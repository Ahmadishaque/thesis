import os

from absl import logging
import ml_collections

import jax.numpy as jnp

import scipy.io
import matplotlib.pyplot as plt

import wandb

from jaxpi.utils import restore_checkpoint
import models

from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, t_star, x_star, y_star, b1, b2, c1, c2, eps1, eps2 = get_dataset(
        config.time_fraction
    )

    # Remove the last time step
    # u_ref = u_ref[:-1, :]
    # v_ref = v_ref[:-1, :]

    u0 = u_ref[0, :, :]
    v0 = v_ref[0, :, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.GreyScott(config, t, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2)

    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        v_star = v_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]

        # Restore the checkpoint
        ckpt_path = os.path.join(
            workdir, config.wandb.name, "ckpt", "time_window_{}".format(idx + 1)
            )

        ckpt_path = os.path.abspath(ckpt_path)

        model.state = restore_checkpoint(model.state, ckpt_path)

        import jax.tree_util as jtu
        # This removes the extra leading dimension (1, ...) if it exists
        model.state = model.state.replace(
            params=jtu.tree_map(lambda x: x[0] if x.ndim > 1 and x.shape[0] == 1 else x, model.state.params)
        )

        # Compute the L2 error for the current time window
        u_error, v_error = model.compute_l2_error(model.state.params, t, x_star, y_star, u_star, v_star)
        print(
            "Time window: {}, u error: {:.3e}, v error: {:.3e}".format(
                idx + 1, u_error, v_error
                )
            )

        # u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
        # v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)

        u_pred = model.u_pred_fn(model.state.params, model.t_star, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(model.state.params, model.t_star, model.x_star, model.y_star)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    # u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    # v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
    u_ref_aligned = u_ref[-u_pred.shape[0]:] 
    v_ref_aligned = v_ref[-v_pred.shape[0]:]

    u_error = jnp.linalg.norm(u_pred - u_ref_aligned) / jnp.linalg.norm(u_ref_aligned)
    v_error = jnp.linalg.norm(v_pred - v_ref_aligned) / jnp.linalg.norm(v_ref_aligned)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))

    # --- PRUNING COMPARISON ---
    import json
    # Point to the summary file downloaded from dense run
    baseline_path = os.path.join(workdir, "dense_baseline.json") 
    
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        dense_u = baseline.get("u_error")
        dense_v = baseline.get("v_error")

        if dense_u and dense_v:
            u_change = ((u_error - dense_u) / dense_u) * 100
            v_change = ((v_error - dense_v) / dense_v) * 100

            logging.info("==========================================")
            logging.info("   NTK-SAP PRUNING ANALYSIS (Ratio: {})".format(config.pruning_ratio))
            logging.info("==========================================")
            logging.info("Dense Baseline U Error: {:.3e}".format(dense_u))
            logging.info("Sparse Current U Error:  {:.3e}".format(u_error))
            logging.info("--> U Accuracy Change:  {:+.2f}%".format(u_change))
            logging.info("------------------------------------------")
            logging.info("Dense Baseline V Error: {:.3e}".format(dense_v))
            logging.info("Sparse Current V Error:  {:.3e}".format(v_error))
            logging.info("--> V Accuracy Change:  {:+.2f}%".format(v_change))
            logging.info("==========================================")
    else:
        logging.warning("Baseline file {} not found. Skipping pruning analysis.".format(baseline_path))
    # ------------------------------------

    # Plot the results
    XX, YY = jnp.meshgrid(x_star, y_star, indexing="ij")  # Grid for plotting

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Plot at the last time step
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, u_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reference u")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, u_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted u")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(u_ref[-1] - u_pred[-1]), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "gs_u.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # Plot at the last time step
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, YY, v_ref[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Reference v")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, YY, v_pred[-1], cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted v")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, YY, jnp.abs(v_ref[-1] - v_pred[-1]), cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "gs_v.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)