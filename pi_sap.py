import jax
import jax.numpy as jnp
from jax import random, tree_util
import models 

def get_ntk_sap_mask(saliency, prune_ratio):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(saliency)
    
    # Identify ONLY the kernel weights for pruning
    scores_to_prune = []
    for path, val in leaves:
        path_str = "".join(str(p) for p in path)
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
            new_leaves.append(jnp.ones_like(val)) # Always keep
            
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

# --- Physics-Informed Saliency logic ---
def compute_ntk_saliency(rng, config, time_window_domain):
    """
    Computes saliency based on the PDE residual (r_net) 
    instead of just the neural net output.
    """
    batch_size = 1000  # [cite: 19]
    num_rounds = 8     # [cite: 18]
    eps = 1e-4         # [cite: 17]
    
    # Setup dummy domain for initialization
    t_star = jnp.linspace(time_window_domain[0], time_window_domain[1], 10)
    x_star = jnp.linspace(-1, 1, 10)
    y_star = jnp.linspace(-1, 1, 10)
    u0, v0 = jnp.zeros((10, 10)), jnp.zeros((10, 10))

    base_model = models.GreyScott(config, t_star, x_star, y_star, u0, v0, 
                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    
    total_saliency = jax.tree_util.tree_map(jnp.zeros_like, base_model.state.params)

    for r in range(num_rounds):
        rng, _, point_key = random.split(rng, 3)
        temp_model = models.GreyScott(config, t_star, x_star, y_star, u0, v0, 
                                     0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
        params = temp_model.state.params
        
        # Sample random space-time coordinates [cite: 19]
        t_rand = random.uniform(point_key, (batch_size,), minval=time_window_domain[0], maxval=time_window_domain[1])
        x_rand = random.uniform(point_key, (batch_size,), minval=-1.0, maxval=1.0)
        y_rand = random.uniform(point_key, (batch_size,), minval=-1.0, maxval=1.0)

        def single_point_pi_saliency(t, x, y):
            flat_params, params_treedef = jax.tree_util.tree_flatten(params)
            had_leading_one = [(p.ndim >= 2 and p.shape[0] == 1) for p in flat_params]
            squeezed_flat = [(jnp.squeeze(p, axis=0) if had else p) for p, had in zip(flat_params, had_leading_one)]
            squeezed_params = jax.tree_util.tree_unflatten(params_treedef, squeezed_flat)

            def pi_proxy_fn(p_squeezed):
                # Physics-Informed: Change in ru and rv residuals 
                ru, rv = temp_model.r_net(p_squeezed, t, x, y)
                
                # Perturb weights
                flat_ps, _ = jax.tree_util.tree_flatten(p_squeezed)
                subkeys = random.split(point_key, len(flat_ps) + 1)[1:]
                noises = [random.normal(k, p.shape) * eps for k, p in zip(subkeys, flat_ps)]
                noise_tree = jax.tree_util.tree_unflatten(params_treedef, noises)
                p_noisy = jax.tree_util.tree_map(lambda a, b: a + b, p_squeezed, noise_tree)

                ru_n, rv_n = temp_model.r_net(p_noisy, t, x, y)
                
                # Saliency = square of the difference in physical residuals 
                return (jnp.sum(ru**2 + rv**2) - jnp.sum(ru_n**2 + rv_n**2))**2 / (eps**2)

            grads_squeezed = jax.grad(pi_proxy_fn)(squeezed_params)
            
            flat_grads_sq, _ = jax.tree_util.tree_flatten(grads_squeezed)
            restored_flat_grads = [(jnp.expand_dims(g, axis=0) if had else g) for g, had in zip(flat_grads_sq, had_leading_one)]
            grads_restored = jax.tree_util.tree_unflatten(params_treedef, restored_flat_grads)

            return jax.tree_util.tree_map(lambda p, g: jnp.abs(p * g), params, grads_restored)

        batch_saliency = jax.vmap(single_point_pi_saliency)(t_rand, x_rand, y_rand)
        round_saliency = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), batch_saliency)
        total_saliency = jax.tree_util.tree_map(lambda x, y: x + y, total_saliency, round_saliency)

    return jax.tree_util.tree_map(lambda x: x / num_rounds, total_saliency)