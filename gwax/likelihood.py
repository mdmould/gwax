import jax
import jax.numpy as jnp


def mean_and_variance(weights, n):
    # mean and variance of the mean
    mean = jnp.sum(weights, axis = -1) / n
    variance = jnp.sum(weights**2, axis = -1) / n**2 - mean**2 / n
    return mean, variance

def ln_mean_and_variance(weights, n):
    # lazy ln(mean) and variance of ln(mean)
    mean, variance = mean_and_variance(weights, n)
    return jnp.log(mean), variance / mean**2

# def ln_mean_and_variance(ln_weights, n):
#     ln_sum = jax.nn.logsumexp(ln_weights, axis = -1)
#     ln_mean = ln_sum - jnp.log(n)
#     ess = jnp.exp(2 * ln_sum - jax.nn.logsumexp(2 * ln_weights, axis = -1))
#     variance = 1 / ess - 1 / n
#     return ln_mean, variance


def ln_likelihood_and_variance(posteriors, injections, model, parameters):
    pe_weights = model(posteriors, parameters) / posteriors['prior'])
    vt_weights = model(injections, parameters) / injections['prior'])
    num_obs, num_pe = pe_weights.shape
    ln_lkls, pe_variances = ln_mean_and_variance(pe_weights, num_pe)
    rate, vt_variance = mean_and_variance(vt_weights, injections['total'])
    ln_lkl = jnp.sum(ln_lkls) - rate * injections['time']
    variance = jnp.sum(pe_variances) + vt_variance * injections['time']**2
    # ln_lkl = jnp.nan_to_num(ln_lkl, nan = -jnp.inf)
    # variance = jnp.nan_to_num(variance, nan = jnp.inf)
    return ln_lkl, variance

