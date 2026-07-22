import jax
import jax.numpy as jnp
import jax_tqdm
import bilby

from .util import logcumsumexp


def estimator(weights, n, axis = -1):
    mean = jnp.sum(weights, axis = axis) / n
    sq_of_mean = mean ** 2
    mean_of_sq = jnp.sum(weights ** 2, axis = axis) / n
    variance = (mean_of_sq - sq_of_mean) / n
    ess = n * sq_of_mean / mean_of_sq
    return mean, variance, ess

def ln_estimator(weights, n, axis = -1):
    mean, variance, ess = estimator(weights, n, axis = axis)
    return jnp.log(mean), variance / mean ** 2, ess

def ln_estimator_from_ln(ln_weights, n, axis = -1):
    ln_sum = jax.nn.logsumexp(ln_weights, axis = axis)
    ln_mean = ln_sum - jnp.log(n)
    ess = jnp.exp(2 * ln_sum - jax.nn.logsumexp(2 * ln_weights, axis = axis))
    variance = 1 / ess - 1 / n
    return ln_mean, variance, ess

def estimator_from_ln(ln_weights, n, axis = -1):
    ln_mean, variance, ess = ln_estimator_from_ln(ln_weights, n, axis = axis)
    return jnp.exp(ln_mean), jnp.exp(jnp.log(variance) + 2 * ln_mean), ess


def estimator_stacked(weights, n):
    idxs = jnp.insert(jnp.cumsum(n), 0, 0)
    weights = jnp.insert(weights, 0, 0)
    sums = jnp.cumsum(weights)
    sums_sq = jnp.cumsum(weights ** 2)
    sums = sums[idxs[1:]] - sums[idxs[:-1]]
    sums_sq = sums_sq[idxs[1:]] - sums_sq[idxs[:-1]]
    means = sums / n
    sq_of_mean = means ** 2
    mean_of_sq = sums_sq / n
    variances = (mean_of_sq - sq_of_mean) / n
    ess = n * sq_of_mean / mean_of_sq
    return means, variances, ess

def ln_estimator_stacked(weights, n):
    means, variances, ess = estimator_stacked(weights, n)
    return jnp.log(means), variances / means ** 2, ess

def ln_estimator_and_variance_stacked_from_ln(ln_weights, n):
    idxs = jnp.insert(jnp.cumsum(n), 0, 0)
    ln_weights = jnp.insert(ln_weights, 0, -jnp.inf)
    ln_sums = logcumsumexp(ln_weights)
    ln_sums_sq = logcumsumexp(2 * ln_weights)
    ln_sums = jax.nn.logsumexp(
        jnp.array([ln_sums[idxs[1:]], ln_sums[idxs[:-1]]]),
        b = jnp.array([1, -1])[:, None],
        axis = 0,
    )
    ln_sums_sq = jax.nn.logsumexp(
        jnp.array([ln_sums_sq[idxs[1:]], ln_sums_sq[idxs[:-1]]]),
        b = jnp.array([1, -1])[:, None],
        axis = 0,
    )
    ln_means = ln_sums - jnp.log(n)
    ess = jnp.exp(2 * ln_sums - ln_sums_sq)
    variances = 1 / ess - 1 / n
    return ln_means, variances, ess

def estimator_and_variance_stacked_from_ln(ln_weights, n):
    ln_means, variances, ess = ln_estimator_stacked_from_ln(ln_weights, n)
    return jnp.exp(ln_means), jnp.exp(jnp.log(variances) + 2 * ln_means), ess


def resample_rate(key, num_obs, vt):
    return jax.random.gamma(key, num_obs, shape = jnp.shape(vt)) / vt

def shape_likelihood_ingredients(posteriors, injections, density, parameters):
    num_obs, total = posteriors['weight'].shape
    pe_weights = density(posteriors, parameters) * posteriors['weight']
    vt_weights = density(injections, parameters) * injections['weight']
    ln_lkls, pe_variances, ess_pe = ln_estimator(pe_weights, total)
    ln_vt, variance_vt, ess_vt = ln_estimator(vt_weights, injections['total'])
    ln_vt += jnp.log(injections['time']) # dependence of variance on T cancels
    variance_pe = pe_variances.sum()
    variance_vt *= num_obs ** 2
    return dict(
        ln_likelihood = ln_lkls.sum() - ln_vt * num_obs,
        ln_vt = ln_vt,
        variance = variance_pe + variance_vt,
        variance_pe = variance_pe,
        variance_vt = variance_vt,
        ess_pe = ess_pe,
        ess_vt = ess_vt,
    )

def rate_likelihood_ingredients(posteriors, injections, density, parameters):
    num_obs, total = posteriors['weight'].shape
    pe_weights = density(posteriors, parameters) * posteriors['weight']
    vt_weights = density(injections, parameters) * injections['weight']
    ln_lkls, pe_variances, ess_pe = ln_estimator(pe_weights, total)
    rate, variance_vt, ess_vt = estimator(vt_weights, injections['total'])
    num = rate * injections['time']
    variance_pe = pe_variances.sum()
    variance_vt *= injections['time'] ** 2
    return dict(
        ln_likelihood = ln_lkls.sum() - num,
        num = num,
        variance = variance_pe + variance_vt,
        variance_pe = variance_pe,
        variance_vt = variance_vt,
        ess_pe = ess_pe,
        ess_vt = ess_vt,
    )


def shape_likelihood_ingredients_stacked(
    posteriors, injections, density, parameters,
):
    pe_weights = density(posteriors, parameters) * posteriors['weight']
    vt_weights = density(injections, parameters) * injections['weight']
    ln_lkls, pe_variances, ess_pe = ln_estimator_stacked(pe_weights, posteriors['total'])
    ln_vt, variance_vt, ess_vt = ln_estimator(vt_weights, injections['total'])
    ln_vt += jnp.log(injections['time']) # dependence of variance on T cancels
    num_obs = posteriors['total'].size
    variance_pe = pe_variances.sum()
    variance_vt *= num_obs ** 2
    return dict(
        ln_likelihood = ln_lkls.sum() - ln_vt * num_obs,
        ln_vt = ln_vt,
        variance = variance_pe + variance_vt,
        variance_pe = variance_pe,
        variance_vt = variance_vt,
        ess_pe = ess_pe,
        ess_vt = ess_vt,
    )

def rate_likelihood_ingredients_stacked(
    posteriors, injections, density, parameters,
):
    pe_weights = density(posteriors, parameters) * posteriors['weight']
    vt_weights = density(injections, parameters) * injections['weight']
    ln_lkls, pe_variances, ess_pe = ln_estimator_stacked(pe_weights, posteriors['total'])
    rate, variance_vt, ess_vt = estimator(vt_weights, injections['total'])
    num = rate * injections['time']
    variance_pe = pe_variances.sum()
    variance_vt *= injections['time'] ** 2
    return dict(
        ln_likelihood = ln_lkls.sum() - num,
        num = num,
        variance = variance_pe + variance_vt,
        variance_pe = variance_pe,
        variance_vt = variance_vt,
        ess_pe = ess_pe,
        ess_vt = ess_vt,
    )


def ln_likelihood(
    likelihood_ingredients,
    maximum_variance,
    posteriors,
    injections,
    density,
    parameters,
):
    ingredients = likelihood_ingredients(
        posteriors, injections, density, parameters,
    )
    ln_lkl = jnp.nan_to_num(
        ingredients['ln_likelihood'], nan = -jnp.inf, posinf = -jnp.inf,
    )
    variance = jnp.nan_to_num(
        ingredients['variance'], nan = jnp.inf, neginf = jnp.inf,
    )
    return jnp.where(variance < maximum_variance, ln_lkl, -jnp.inf)

class BilbyLikelihood(bilby.Likelihood):
    def __init__(
        self,
        likelihood_ingredients,
        maximum_variance,
        posteriors,
        injections,
        density,
    ):
        super().__init__()
        self.num_obs = posteriors['total'].size
        self.posteriors = posteriors
        self.injections = injections
        self.maximum_variance = maximum_variance

        self._log_likelihood = \
            lambda posteriors, injections, parameters: ln_likelihood(
                likelihood_ingredients,
                maximum_variance,
                posteriors,
                injections,
                density,
                parameters,
            )

        self._likelihood_ingredients = \
            lambda posteriors, injections, parameters: likelihood_ingredients(
                posteriors, injections, density, parameters,
            )

    def log_likelihood(self, parameters):
        return jax.jit(self._log_likelihood)(
            self.posteriors, self.injections, parameters,
        )

    def likelihood_ingredients(self, parameters):
        return jax.jit(self._likelihood_ingredients)(
            self.posteriors, self.injections, parameters,
        )

def postprocess_bilby(result, likelihood):
    n = len(result.posterior)
    posterior = {k: jnp.array(v) for k, v in result.posterior.items()}

    @jax_tqdm.scan_tqdm(n, print_rate = 1, tqdm_type = 'std')
    def single(carry, x):
        i, parameters = x
        return carry, likelihood.likelihood_ingredients(parameters)

    ingredients = jax.lax.scan(single, None, (jnp.arange(n), posterior))[1]

    if 'ln_vt' in ingredients:
        ingredients['rate'] = resample_rate(
            jax.random.key(0),
            likelihood.num_obs,
            jnp.exp(ingredients['ln_vt']),
        )

    for k in ingredients:
        result.posterior[k] = ingredients[k]

    return result

def prior_fraction(likelihood, priors, n = 10_000):
    samples = priors.sample(n)
    for k in samples:
        samples[k] = jnp.array(samples[k])

    @jax_tqdm.scan_tqdm(n, print_rate = 1, tqdm_type = 'std')
    @jax.jit
    def single(carry, x):
        i, parameters = x
        return carry, likelihood.likelihood_ingredients(parameters)['variance']

    variances = jax.lax.scan(single, None, (jnp.arange(n), samples))[1]
    w = variances < likelihood.maximum_variance
    frac = w.mean()
    error = ((jnp.mean(w**2) - frac**2) / n)**0.5
    # error = (frac * (1 - frac) / n)**0.5

    return frac, error

def evidence(result, likelihood, priors, n = 10_000):
    fraction, fraction_error = prior_fraction(likelihood, priors, n)
    fraction = float(fraction)
    fraction_error = float(fraction_error)
    return dict(
        prior_fraction = fraction,
        prior_fraction_error = fraction_error,
        ln_evidence_bilby = result.log_evidence,
        ln_evidence_bilby_error = result.log_evidence_err,
        ln_evidence = result.log_evidence + float(jnp.log(fraction)),
        ln_evidence_error = (
            result.log_evidence_err ** 2 + (fraction_error / fraction) ** 2
        ) ** 0.5,
    )
