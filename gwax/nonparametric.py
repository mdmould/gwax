import jax
import jax.numpy as jnp


def get_adjacent(*shape):
    grid = jnp.arange(jnp.array(shape).prod()).reshape(shape)
    pairs_list = []
    for axis in range(len(shape)):
        sl1 = [slice(None)] * len(shape)
        sl2 = [slice(None)] * len(shape)
        sl1[axis] = slice(0, -1)
        sl2[axis] = slice(1, None)
        a = grid[tuple(sl1)].ravel()
        b = grid[tuple(sl2)].ravel()
        pairs_list.append(jnp.stack([a, b], axis = 1))
    return jnp.vstack(pairs_list)

def get_bins(samples, edges):
    multi_index = [
        jnp.clip(jnp.digitize(x, e) - 1, 0, e.size - 2)
        for x, e in zip(samples, edges)
    ]
    dims = [e.size - 1 for e in edges]
    return jnp.ravel_multi_index(multi_index, dims, mode = 'clip')

def get_bins_1d(samples, edges):
    return jnp.clip(jnp.digitize(samples, edges) - 1, 0, len(edges) - 2)

def icar_penalty(adj, y):
    return jnp.sum(jnp.diff(y[adj], axis = 1) ** 2) / 2

def ln_prior_icar(n, adj, y, tau):
    penalty = icar_penalty(adj, y)
    ln_prior = jnp.log(tau) * (n - 1) / 2 - penalty * tau
    return ln_prior

def ln_prior_icar_gamma(n, adj, y, a, b):
    penalty = icar_penalty(adj, y)
    ln_prior = -(a + (n - 1) / 2) * jnp.log(b + penalty)
    return ln_prior

def resample_tau(key, adj, y, a, b):
    *shape, n = y.shape
    penalty = jax.vmap(lambda y: icar_penalty(adj, y))(y.reshape(-1, n))
    gs = jax.random.gamma(key, a + (n - 1) / 2, penalty.shape)
    tau = gs / (b + penalty)
    return tau.reshape(shape)
