import jax
import jax.numpy as jnp
import jax_tqdm
import equinox


def trainer(
    key,
    params,
    loss_fn,
    optimizer,
    steps,
    print_rate = 1,
):
    @jax_tqdm.scan_tqdm(steps, print_rate = print_rate)
    @equinox.filter_jit
    def update(carry, step):
        key, params, state, best_loss, best_params = carry
        new_key, key = jax.random.split(key)
        loss, grad = equinox.filter_value_and_grad(loss_fn)(params, key, step)
        updates, state = optimizer.update(grad, state, params)
        new_params = equinox.apply_updates(params, updates)
        best_loss, best_params = jax.lax.cond(
            loss < best_loss,
            lambda: (loss, params),
            lambda: (best_loss, best_params),
        )
        carry = new_key, new_params, state, best_loss, best_params
        return carry, loss

    (key, last_params, state, best_loss, best_params), losses = jax.lax.scan(
        update,
        (key, params, optimizer.init(params), jnp.inf, params),
        jnp.arange(steps),
    )
    
    return losses, last_params, best_params

