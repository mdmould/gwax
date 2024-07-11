import jax
import jax.numpy as jnp
import jax_tqdm
import equinox


@equinox.filter_jit
def update(params, loss_fn, optimizer, state, *args):
    loss_and_grad = equinox.filter_value_and_grad(loss_fn, has_aux = True)
    (loss, *args), grad = loss_and_grad(params, *args)
    updates, state = optimizer.update(grad, state, params)
    params = equinox.apply_updates(params, updates)
    return params, state, loss, *args


def trainer(
    params,
    loss_fn,
    optimizer,
    steps,
    *args,
    print_rate = 1,
):
    def _update(params, state, *args):
        return update(params, loss_fn, optimizer, state, *args)

    @jax_tqdm.scan_tqdm(steps, print_rate = print_rate)
    @equinox.filter_jit
    def loop(carry, step):
        params, state, best_loss, best_params, *args = carry
        new_params, state, loss, *args = _update(params, state, *args)
        best_loss, best_params = jax.lax.cond(
            loss < best_loss,
            lambda: (loss, params),
            lambda: (best_loss, best_params),
        )
        return (new_params, state, best_loss, best_params, *args), loss

    (params, state, best_loss, best_params, *args), losses = jax.lax.scan(
        loop,
        (params, optimizer.init(params), jnp.inf, params, *args),
        jnp.arange(steps),
    )
    
    return losses, params, best_params

