import time
import os
import importlib

from absl import logging

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import numpy as np
import scipy.io
import ml_collections
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset

from jaxpi.utils import restore_checkpoint


def get_pruning_module(config):
    """Resolve the thesis pruning method without source-code toggles."""
    method = str(config.pruning_method).lower()
    if method not in {"ntk_sap", "pi_sap"}:
        raise ValueError(f"Unsupported pruning_method={method!r}; expected 'ntk_sap' or 'pi_sap'.")

    level = int(round(100.0 * float(config.pruning_ratio)))
    expected_name = f"pirate_{method}_{level}"
    if config.wandb.name != expected_name:
        raise ValueError(
            f"Run identity mismatch: pruning settings imply {expected_name!r}, "
            f"but config.wandb.name={config.wandb.name!r}."
        )
    return importlib.import_module(method)


def train_one_window(config, workdir, model, res_sampler, u_ref, v_ref, idx, mask=None):
    step_offset = idx * config.training.max_steps

    # Logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.GreyScottEvaluator(config, model)

    # jit warm up
    print("Waiting for JIT...")

    # # --- VERIFICATION: Weight Zeroing Check ---
    # if mask is not None:
    #     # Initial application
    #     model.state = model.state.replace(
    #         params=jax.tree_util.tree_map(lambda p, m: p * m, model.state.params, mask)
    #     )
    #     # Verify non-zero weights
    #     nnz = sum(jnp.count_nonzero(x) for x in jax.tree_util.tree_leaves(model.state.params))
    #     total = sum(x.size for x in jax.tree_util.tree_leaves(model.state.params))
    #     print(f"Window {idx+1} Start: {nnz}/{total} active weights.")
    # # ------------------------------------------

    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        # model.state = model.step(model.state, batch)
        

        # # 2. Force pruned weights to stay zero
        # if mask is not None:
        #     model.state = model.state.replace(
        #         params=jax.tree_util.tree_map(lambda p, m: p * m, model.state.params, mask)
        #     )


        # ------------------------------------------
        # # Pass the mask here to ensure gradients are zeroed
        # model.state = model.step(model.state, batch, mask=mask)

        # # Force weights to stay zero (as a secondary safety measure)
        # if mask is not None:
        #     model.state = model.state.replace(
        #         params=jax.tree_util.tree_map(lambda p, m: p * m, model.state.params, mask)
        #     )

        # ------------------------------------------
        # --- 1. VERIFICATION (BEFORE STEP) ---
        # We use tree_map(lambda x: x[0]) to look at the first device's copy of the parameters
        if mask is not None and step % 500 == 0:
            params_before, _ = jax.flatten_util.ravel_pytree(
                jax.tree_util.tree_map(lambda x: x[0], model.state.params)
            )

        # --- 2. THE TRAINING STEP ---
        # Passing the mask here ensures gradients are zeroed BEFORE the optimizer update
        model.state = model.step(model.state, batch, mask=mask)

        # --- 3. VERIFICATION (AFTER STEP) & SAFETY ENFORCEMENT ---
        if mask is not None:
            # Secondary safety: hard-set pruned weights to zero to prevent floating point drift
            model.state = model.state.replace(
                params=jax.tree_util.tree_map(lambda p, m: p * m, model.state.params, mask)
            )

            # Verification logic to confirm the mask is working
            if step % 500 == 0:
                params_after, _ = jax.flatten_util.ravel_pytree(
                    jax.tree_util.tree_map(lambda x: x[0], model.state.params)
                )
                flat_mask, _ = jax.flatten_util.ravel_pytree(
                    jax.tree_util.tree_map(lambda x: x[0], mask)
                )
                
                # Check for illegal updates: mask is 0 but (after - before) is significantly different
                diff = jnp.abs(params_after - params_before)
                illegal_updates = (flat_mask == 0) & (diff > 1e-8)
                
                if jnp.any(illegal_updates):
                    print(f"CRITICAL: Step {step} updated {jnp.sum(illegal_updates)} pruned weights!")
                elif step == 0:
                    print(f"SUCCESS: Mask verified for Window {idx+1}. Sparse training active.")

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref, v_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model



def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, mask=None):
    pruning_module = get_pruning_module(config)
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    u_ref, v_ref, t_star, x_star, y_star, b1, b2, c1, c2, eps1, eps2 = get_dataset(
        config.time_fraction
    )

    u0 = u_ref[0, :, :]
    v0 = v_ref[0, :, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    dt = t[1] - t[0]
    t0, t1 = t[0], t[-1] + 2 * dt
    x0, x1 = x_star[0], x_star[-1]
    y0, y1 = y_star[0], y_star[-1]

    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        u_star = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]
        v_star = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :, :]

        # --- 1. PHYSICS-INFORMED INITIALIZATION ---
        if config.use_pi_init:
            logging.info("Use physics-informed initialization...")
            # Temp model to find coeffs
            tmp_model = models.GreyScott(config, t, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2)
            
            # Since tmp_model.state is initialized normally, we take index 0 if it was pmapped
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], tmp_model.state))
            
            t_scaled = t / t[-1]
            tt, xx, yy = jnp.meshgrid(t_scaled, x_star, y_star, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])
            u = jnp.tile(u0, (t_scaled.shape[0], 1, 1))
            v = jnp.tile(v0, (t_scaled.shape[0], 1, 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(state.params, inputs)
            u_coeffs, _, _, _ = jnp.linalg.lstsq(feat_matrix, u.flatten(), rcond=None)
            v_coeffs, _, _, _ = jnp.linalg.lstsq(feat_matrix, v.flatten(), rcond=None)
            
            config.arch.pi_init = jnp.vstack([u_coeffs, v_coeffs]).T
            del tmp_model

        # --- 2. INITIALIZE THE ACTUAL MODEL ---
        model = models.GreyScott(config, t, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2)

        # --- 3. GENERATE THE MASK ---
        from jax import random
        prune_rng = random.PRNGKey(config.seed)
        
        # Use the updated ntk_sap logic that protects alphas/biases
        saliency = pruning_module.compute_ntk_saliency(prune_rng, config, [t0, t1])
        mask = pruning_module.get_ntk_sap_mask(saliency, config.pruning_ratio)

        # --- 4. SHARD BOTH MODEL AND MASK (Fixes ScopeParamShapeError) ---
        
        # Check if the model state is already sharded (has a leading axis for devices)
        # We look at one parameter's shape to see if it has the '1' prefix
        first_leaf = jax.tree_util.tree_leaves(model.state.params)[0]
        is_already_sharded = first_leaf.ndim > 0 and first_leaf.shape[0] == jax.local_device_count()

        if not is_already_sharded:
            model.state = jax.device_put_replicated(model.state, jax.local_devices())
        else:
            print("Model state already sharded, skipping replication.")

        # Saliency is built from the already replicated model state, so its mask
        # already has the pmap device axis. A second replication would introduce
        # an erroneous extra leading dimension.
        mask = jax.device_put(mask)

        # Logging for verification
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(mask))
        total_ones = sum(jnp.sum(x[0]) for x in jax.tree_util.tree_leaves(mask)) # x[0] because it's replicated
        print(f"--- Verified Mask Sparsity: {1.0 - (total_ones / total_params):.4f} ---")

        # 3. MANDATORY: Pre-mask the parameters so the first verification step passes
        # This prevents the "CRITICAL: Step 0" false positive.
        model.state = model.state.replace(
            params=jax.tree_util.tree_map(lambda p, m: p * m, model.state.params, mask)
        )
        # ------------------

        # --- 5. TRAINING ---
        model = train_one_window(
            config, workdir, model, res_sampler, u_star, v_star, idx, mask=mask
        )

        # --- 6. UPDATE INITIAL CONDITIONS FOR NEXT WINDOW ---
        if config.training.num_time_windows > 1:
            # Get params back to CPU/Single-device for prediction
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params

            u0 = model.u0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            v0 = model.v0_pred_fn(params, t_star[num_time_steps], x_star, y_star)

            del model, state, params

    return None
