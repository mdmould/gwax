import jax
import jax.numpy as jnp
import numpy as np
import numpyro


def keep_idx_map(keep):
    return np.cumsum(keep.ravel()) - 1
    # return np.clip(np.cumsum(keep) - 1, 0, np.sum(keep) - 1)

def get_bins_1d(samples, edges):
    return np.clip(np.digitize(samples, edges) - 1, 0, np.size(edges) - 2)

def get_bins(samples, edges, keep = None):
    multi_index = [get_bins_1d(s, e) for s, e in zip(samples, edges)]
    shape = tuple(len(e) - 1 for e in edges)
    idx = np.ravel_multi_index(multi_index, shape)
    if keep is not None:
        assert np.shape(keep) == shape
        keep = keep.ravel()
        assert keep[idx].all()
        idx = keep_idx_map(keep)[idx]
    return idx

def get_adjacent(shape, keep = None):
    idx = np.arange(np.prod(shape)).reshape(shape)
    adjacent = []
    for axis in range(len(shape)):
        s1 = [slice(None)] * len(shape)
        s2 = [slice(None)] * len(shape)
        s1[axis] = slice(None, -1)
        s2[axis] = slice(1, None)
        i = idx[tuple(s1)].ravel()
        j = idx[tuple(s2)].ravel()
        pairs = np.stack([i, j], axis = 1)
        adjacent.append(pairs)
    adjacent = np.concatenate(adjacent)

    if keep is not None:
        assert np.shape(keep) == shape
        keep = keep.ravel()
        adjacent = adjacent[keep[adjacent].all(axis = 1)]
        adjacent = keep_idx_map(keep)[adjacent]

    return adjacent


# original function space sampling

def improper_sample(name, n = None, vol = None):
    if vol is None:
        y = numpyro.sample(
            name,
            numpyro.distributions.ImproperUniform(
                numpyro.distributions.constraints.real, (), (n,),
            ),
        )
    else:
        y = numpyro.sample(
            f'_{name}',
            numpyro.distributions.ImproperUniform(
                numpyro.distributions.constraints.zero_sum(), (), (vol.size,),
            ),
        )
        y -= jax.nn.logsumexp(y + jnp.log(vol))
        y = numpyro.deterministic(name, y)
    return y

# def improper_sample(name, n, keep = None):
#     y = numpyro.sample(
#         name,
#         numpyro.distributions.ImproperUniform(
#             numpyro.distributions.constraints.real, (), (n,),
#         ),
#     )
#     if keep is not None:
#         y = jnp.full_like(keep.astype(float), -jnp.inf).at[keep].set(y)
#     return y

# def improper_sample_norm(name, n, vol, keep = None):
#     y = improper_sample(f'_{name}', n, keep)
#     y -= jax.nn.logsumexp(y + jnp.log(vol))
#     return numpyro.deterministic(name, y)

# def _improper_sample_norm(name, n, vol, keep = None):
#     y = numpyro.sample(
#         name,
#         numpyro.distributions.ImproperUniform(
#             numpyro.distributions.constraints.real, (), (n - 1,),
#         ),
#     )
#     y = jnp.insert(y, -1, -jnp.sum(y))
#     if keep is not None:
#         y = jnp.full_like(keep.astype(float), -jnp.inf).at[keep].set(y)
#     y -= jax.nn.logsumexp(y + jnp.log(vol))
#     return y

def icar_rv(adj, y):
    return jnp.diff(y[adj], axis = 1).squeeze()

def icar_penalty(adj, y):
    return jnp.sum(icar_rv(adj, y) ** 2) / 2

def ln_prior_icar(adj, y, tau):
    penalty = icar_penalty(adj, y)
    ln_prior = jnp.log(tau) * (y.size - 1) / 2 - penalty * tau
    return ln_prior

def ln_prior_icar_gamma(adj, y, a, b):
    penalty = icar_penalty(adj, y)
    ln_prior = -(a + (y.size - 1) / 2) * jnp.log(b + penalty)
    return ln_prior

def resample_tau(key, adj, y, a, b):
    *shape, n = y.shape
    penalty = jax.vmap(lambda y: icar_penalty(adj, y))(y.reshape(-1, n))
    gs = jax.random.gamma(key, a + (n - 1) / 2, penalty.shape)
    tau = gs / (b + penalty)
    return tau.reshape(shape)

def ln_prior_icar_1d_t(adj, y, sigma, nu):
    t = icar_rv(adj, y)
    ln_prior = jax.scipy.stats.t.logpdf(t, nu, scale = sigma).sum()
    # ln_prior = jax.scipy.stats.t.logpdf(t, nu).sum()
    # y *= sigma
    return ln_prior

def ln_prior_icar_1d_multivariate_t(adj, y, sigma, nu):
    t = icar_rv(adj, y)
    tril = jnp.eye(y.size - 1) * sigma
    dist = numpyro.distributions.MultivariateStudentT(nu, scale_tril = tril)
    ln_prior = dist.log_prob(t)
    return ln_prior


# reparametrized basis sampling

def ln_prior_whitened(z, tau):
    return jnp.log(tau) * z.size / 2 - jnp.sum(z ** 2) * tau / 2

def ln_prior_whitened_gamma(z, a = 0, b = 0):
    return -(a + z.size / 2) * jnp.log(b + jnp.sum(z ** 2) / 2)

def regular_graph_laplacian_eigenvalues(shape):
    eigenvalues = np.zeros(shape)
    for axis, n in enumerate(shape):
        k = np.arange(n)
        axis_eigenvalues = 2 - 2 * np.cos(np.pi * k / n)
        reshape = [1] * len(shape)
        reshape[axis] = n
        eigenvalues += axis_eigenvalues.reshape(reshape)
    return eigenvalues

def improper_sample_unwhitened_regular_graph(
    name, eigenvalues, vol = None, ln_prior = ln_prior_whitened_gamma,
):
    n = eigenvalues.size

    z = numpyro.sample(
        f'_{name}',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real, (), (n - 1,),
        ),
    )
    numpyro.factor(f'ln_prior_{name}', ln_prior(z))

    if vol is None:
        mu = numpyro.sample(
            f'_mu_{name}',
            numpyro.distributions.ImproperUniform(
                numpyro.distributions.constraints.real, (), (),
            ),
        )
        c0 = mu * n ** 0.5
    else:
        c0 = 0

    c = jnp.insert(z / eigenvalues.ravel()[1:] ** 0.5, 0, c0)
    c = c.reshape(eigenvalues.shape)
    y = jax.scipy.fft.idctn(c, type = 2, norm = 'ortho')
    y = y.ravel()

    if vol is not None:
        y -= jax.nn.logsumexp(y + jnp.log(vol))

    return numpyro.deterministic(name, y)

def graph_laplacian(shape, keep = None):
    adjacent = get_adjacent(shape, keep)
    n = np.prod(shape) if keep is None else keep.sum()
    laplacian = np.zeros((n, n))
    i, j = adjacent.T
    np.add.at(laplacian, (i, i), 1)
    np.add.at(laplacian, (j, j), 1)
    laplacian[i, j] = -1
    laplacian[j, i] = -1
    return laplacian

def unwhiten_icar(laplacian):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    # select = eigenvalues > 1e-10
    # eigenvalues = eigenvalues[select]
    # eigenvectors = eigenvectors[:, select]
    unwhiten = eigenvectors / eigenvalues[None, :] ** 0.5
    return unwhiten

def improper_sample_unwhitened(
    name, unwhiten, vol = None, ln_prior = ln_prior_whitened_gamma,
):
    n, m = unwhiten.shape
    assert n == m

    z = numpyro.sample(
        f'_{name}',
        numpyro.distributions.ImproperUniform(
            numpyro.distributions.constraints.real, (), (n - 1,),
        ),
    )
    numpyro.factor(f'ln_prior_{name}', ln_prior(z))

    y = unwhiten[:, 1:] @ z

    if vol is None:
        mu = numpyro.sample(
            f'_mu_{name}',
            numpyro.distributions.ImproperUniform(
                numpyro.distributions.constraints.real, (), (),
            ),
        )
        y += mu
    else:
        y -= jax.nn.logsumexp(y + jnp.log(vol))

    return numpyro.deterministic(name, y)
