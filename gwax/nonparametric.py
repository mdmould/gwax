import jax
import jax.numpy as jnp
import numpy as np


def get_adjacent(*shape):
    grid = np.arange(np.prod(shape)).reshape(shape)
    pairs_list = []
    for axis in range(len(shape)):
        sl1 = [slice(None)] * len(shape)
        sl2 = [slice(None)] * len(shape)
        sl1[axis] = slice(0, -1)
        sl2[axis] = slice(1, None)
        a = grid[tuple(sl1)].ravel()
        b = grid[tuple(sl2)].ravel()
        pairs_list.append(np.stack([a, b], axis=1))
    return jnp.vstack(pairs_list)
    
def get_bins_1d(samples, edges):
    return jnp.clip(jnp.digitize(samples, edges) - 1, 0, len(edges) - 2)

def get_bins_nd(samples, edges):
    multi_index = [
        jnp.clip(jnp.digitize(x, e) - 1, 0, len(e) - 2)
        for x, e in zip(samples, edges)
    ]
    dims = [len(e) - 1 for e in edges]
    return jnp.ravel_multi_index(multi_index, dims)

def icar_penalty(adj, y):
    return jnp.sum(jnp.diff(y[adj], axis = 1).squeeze() ** 2) / 2

def ln_prior_icar(adj, y, tau):
    penalty = icar_penalty(adj, y)
    ln_prior = jnp.log(tau) * (y.size - 1) / 2 - penalty * tau
    return ln_prior

def ln_prior_icar_gamma_marginalized_tau(adj, y, a, b):
    penalty = icar_penalty(adj, y)
    ln_prior = -(a + (y.size - 1) / 2) * jnp.log(b + penalty)
    return ln_prior

def resample_tau(key, adj, y, a, b):
    penalty = icar_penalty(adj, y)
    g = jax.random.gamma(key, a + (y.size - 1) / 2)
    tau = g / (b + penalty)
    return tau

def resample_taus(key, adj, ys, a, b):
    penalties = jax.vmap(lambda y: icar_penalty(adj, y))(ys)
    gs = jax.random.gamma(key, a + (y.shape[-1] - 1) / 2, y.shape[:-1])
    taus = gs / (b + penalties)
    return taus
