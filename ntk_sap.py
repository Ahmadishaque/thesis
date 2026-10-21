import jax
import jax.numpy as jnp
from jax import random, tree_util

from typing import Any, Callable, Dict, Tuple
import models  # Your existing PirateNet/Grey-Scott model file

# --- Tree Helpers ---
def tree_mul(a, b):
    return jax.tree_util.tree_map(lambda x, y: x * y, a, b)

def tree_ones_like(pytree):
    return jax.tree_util.tree_map(jnp.ones_like, pytree)

# --- Core NTK-SAP Logic ---
def compute_ntk_saliency(rng, config, time_window_domain):
    # Setup dummy domain for pruning (Window 1 range)
    # Sampling 1000 random points for NTK proxy calculation
    batch_size = 1000
    num_rounds = 8 
    eps = 1e-4
    
    # We need dummy data to initialize the temporary models
    # These won't be used for training, just to get the weight structure
    t_star = jnp.linspace(time_window_domain[0], time_window_domain[1], 10)
    x_star = jnp.linspace(-1, 1, 10)
    y_star = jnp.linspace(-1, 1, 10)
    u0 = jnp.zeros((10, 10))
    v0 = jnp.zeros((10, 10))

    # Initialize a dummy model to get the structure of parameters
    base_model = models.GreyScott(config, t_star, x_star, y_star, u0, v0, 
                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    
    total_saliency = jax.tree_util.tree_map(jnp.zeros_like, base_model.state.params)

    for r in range(num_rounds):
        rng, weight_key, point_key = random.split(rng, 3)

        temp_model = models.GreyScott(
            config, t_star, x_star, y_star, u0, v0,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01
        )
        params = temp_model.state.params
        
        # 2. Sample random inputs
        t_rand = random.uniform(point_key, (batch_size,), minval=time_window_domain[0], maxval=time_window_domain[1])
        x_rand = random.uniform(point_key, (batch_size,), minval=-1.0, maxval=1.0)
        y_rand = random.uniform(point_key, (batch_size,), minval=-1.0, maxval=1.0)

        # 3. Define the saliency per single point
        def single_point_saliency(t, x, y):
            # Flatten original params and record which leaves had leading singleton axis
            flat_params, params_treedef = jax.tree_util.tree_flatten(params)
            had_leading_one = [(p.ndim >= 2 and p.shape[0] == 1) for p in flat_params]

            # Build squeezed flat params (remove leading axis where present)
            squeezed_flat = [
                (jnp.squeeze(p, axis=0) if had else p) for p, had in zip(flat_params, had_leading_one)
            ]
            squeezed_params = jax.tree_util.tree_unflatten(params_treedef, squeezed_flat)

            # If any had leading-one, we likely need batch-shaped inputs for forward; otherwise scalars ok
            need_batch_input = any(had_leading_one)
            if need_batch_input:
                t_in = jnp.expand_dims(t, 0)
                x_in = jnp.expand_dims(x, 0)
                y_in = jnp.expand_dims(y, 0)
            else:
                t_in, x_in, y_in = t, x, y

            # Use grads wrt squeezed params
            def ntk_proxy_fn(p_squeezed):
                # forward using squeezed params (shapes match what apply expects)
                u, v = temp_model.neural_net(p_squeezed, t_in, x_in, y_in)

                # build per-leaf noise matching the squeezed param shapes
                flat_p_squeezed, _ = jax.tree_util.tree_flatten(p_squeezed)
                n_leaves = len(flat_p_squeezed)
                # split point_key deterministically for leaves (we keep using same point_key per vmap element)
                subkeys = random.split(point_key, n_leaves + 1)[1:]
                noises = [random.normal(k, p.shape) * eps for k, p in zip(subkeys, flat_p_squeezed)]
                noise_tree = jax.tree_util.tree_unflatten(params_treedef, noises)

                p_noisy = jax.tree_util.tree_map(lambda a, b: a + b, p_squeezed, noise_tree)

                u_n, v_n = temp_model.neural_net(p_noisy, t_in, x_in, y_in)
                return jnp.sum((u - u_n)**2 + (v - v_n)**2) / (eps**2)

            # grads w.r.t squeezed params
            grads_squeezed = jax.grad(ntk_proxy_fn)(squeezed_params)

            # restore gradient shapes to original (expand leading axis where it was removed)
            flat_grads_squeezed, _ = jax.tree_util.tree_flatten(grads_squeezed)
            restored_flat_grads = [
                (jnp.expand_dims(g, axis=0) if had else g) for g, had in zip(flat_grads_squeezed, had_leading_one)
            ]
            grads_restored = jax.tree_util.tree_unflatten(params_treedef, restored_flat_grads)

            # Saliency computed on original params & restored grads (shapes match)
            saliency = jax.tree_util.tree_map(lambda p, g: jnp.abs(p * g), params, grads_restored)
            return saliency

        # 4. Use vmap to compute saliency across the batch and then average
        # This keeps internal Fourier shapes (3,) and avoids the Scope Error
        batch_saliency = jax.vmap(single_point_saliency)(t_rand, x_rand, y_rand)
        round_saliency = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), batch_saliency)
        
        total_saliency = jax.tree_util.tree_map(lambda x, y: x + y, total_saliency, round_saliency)

    return jax.tree_util.tree_map(lambda x: x / num_rounds, total_saliency)

def get_ntk_sap_mask(saliency, prune_ratio):
    # Flatten with path to identify specifically named parameters
    leaves, treedef = jax.tree_util.tree_flatten_with_path(saliency)
    
    # Identify ONLY the kernel weights for pruning
    scores_to_prune = []
    for path, val in leaves:
        path_str = "".join(str(p) for p in path)
        # We only consider 'kernel' weights for the 90% budget
        # We EXCLUDE 'alpha' even if they are in a kernel path
        if 'kernel' in path_str and 'alpha' not in path_str:
            scores_to_prune.append(val.reshape(-1))
    
    all_scores = jnp.concatenate(scores_to_prune)
    threshold = jnp.percentile(all_scores, prune_ratio * 100)
    
    new_leaves = []
    for path, val in leaves:
        path_str = "".join(str(p) for p in path)
        # Only prune kernels. PROTECT alpha, biases, and pi_init coefficients.
        if 'kernel' in path_str and 'alpha' not in path_str:
            new_leaves.append((val > threshold).astype(jnp.float32))
        else:
            # This is the "Safety Net": alphas and biases are always 1.0
            new_leaves.append(jnp.ones_like(val)) 
            
    return jax.tree_util.tree_unflatten(treedef, new_leaves)