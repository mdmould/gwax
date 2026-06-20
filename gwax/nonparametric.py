import jax
import jax.numpy as jnp
import numpyro


def get_bins(samples, edges):
    multi_index = [
        jnp.clip(jnp.digitize(x, e) - 1, 0, e.size - 2)
        for x, e in zip(samples, edges)
    ]
    dims = [e.size - 1 for e in edges]
    return jnp.ravel_multi_index(multi_index, dims, mode = 'clip')

def get_bins_1d(samples, edges):
    return jnp.clip(jnp.digitize(samples, edges) - 1, 0, len(edges) - 2)


## TODO: make adj refer to nodes after filtering by keep
def get_adjacent(*shape, keep = None):
    grid = jnp.arange(jnp.array(shape).prod()).reshape(shape)
    adj = []
    for axis in range(len(shape)):
        sl1 = [slice(None)] * len(shape)
        sl2 = [slice(None)] * len(shape)
        sl1[axis] = slice(0, -1)
        sl2[axis] = slice(1, None)
        a = grid[tuple(sl1)].ravel()
        b = grid[tuple(sl2)].ravel()
        adj.append(jnp.stack([a, b], axis = 1))
    adj = jnp.vstack(adj)
    if keep is not None:
        adj = adj[keep[adj].all(axis = 1)]
    return adj


## TODO: make adj refer to nodes after filtering by keep
def improper_sample(key, n, adj, keep = None):
    y = numpyro.sample(
        f'_{key}',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real, (), (n,),
        ),
    )
    if keep is not None:
        y = jnp.full_like(keep.astype(float), -jnp.inf).at[keep].set(y)
    return y

def improper_sample_norm(key, n, adj, vol, keep = None):
    y = improper_sample(key, n, adj, keep)
    y -= jax.nn.logsumexp(y + jnp.log(vol))
    return y

def _improper_sample_norm(key, n, adj, vol, keep = None):
    y = numpyro.sample(
        f'_{key}',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real, (), (n - 1,),
        ),
    )
    y = jnp.insert(y, -1, -jnp.sum(y))
    if keep is not None:
        y = jnp.full_like(keep.astype(float), -jnp.inf).at[keep].set(y)
    y -= jax.nn.logsumexp(y + jnp.log(vol))
    return y


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


def ln_prior_icar_t(adj, y, tau, nu):
    
