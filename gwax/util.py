import jax.numpy as jnp


def logcumsumexp(x, axis = None):
    c = jnp.max(x)
    return c + jnp.log(jnp.cumsum(jnp.exp(x - c), axis = axis))

def cumulative_trapezoid(y, x):
    return jnp.insert(jnp.cumsum((y[1:] + y[:-1]) * jnp.diff(x) / 2), 0, 0)
