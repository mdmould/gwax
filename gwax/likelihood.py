import jax
import jax.numpy as jnp
import bilby


def estimator_and_variance(weights, n, axis = None):
    # mean and variance of the mean
    mean = jnp.sum(weights, axis = axis) / n
    variance = jnp.sum(weights ** 2, axis = axis) / n ** 2 - mean ** 2 / n
    return mean, variance

def ln_estimator_and_variance(weights, n, axis = None):
    # lazy ln(mean) and variance of ln(mean)
    mean, variance = estimator_and_variance(weights, n, axis = axis)
    return jnp.log(mean), variance / mean ** 2

# def ln_estimator_and_variance(ln_weights, n):
#     ln_sum = jax.nn.logsumexp(ln_weights, axis = -1)
#     ln_mean = ln_sum - jnp.log(n)
#     ess = jnp.exp(2 * ln_sum - jax.nn.logsumexp(2 * ln_weights, axis = -1))
#     variance = 1 / ess - 1 / n
#     return ln_mean, variance


def shape_likelihood_ingredients(posteriors, injections, density, parameters):
    pe_weights = density(posteriors, parameters) / posteriors['prior']
    vt_weights = density(injections, parameters) / injections['prior']
    num_obs, num_pe = pe_weights.shape
    ln_lkls, pe_variances = ln_estimator_and_variance(pe_weights, num_pe, axis = -1)
    pe_variance = jnp.sum(pe_variances)
    ln_volume, vt_variance = ln_estimator_and_variance(vt_weights, injections['total'])
    ln_vt = ln_volume + jnp.log(injections['time']) # dependence of variance on T cancels
    ln_lkl = jnp.sum(ln_lkls) - ln_vt * num_obs
    variance = pe_variance + vt_variance * num_obs ** 2
    return dict(
        ln_likelihood = ln_lkl,
        variance = variance,
        pe_variance = pe_variance,
        vt_variance = vt_variance,
        ln_vt = ln_vt,
    )

def rate_likelihood_ingredients(posteriors, injections, density, parameters):
    num_obs, num_pe = posteriors['prior'].shape
    pe_weights = density(posteriors, parameters) / posteriors['prior']
    vt_weights = density(injections, parameters) / injections['prior']
    ln_lkls, pe_variances = ln_estimator_and_variance(pe_weights, num_pe, axis = -1)
    rate, vt_variance = estimator_and_variance(vt_weights, injections['total'])
    num_exp = rate * injections['time']
    ln_lkl = jnp.sum(ln_lkls) - num_exp
    pe_variance = jnp.sum(pe_variances)
    variance = pe_variance + vt_variance * injections['time'] ** 2
    return dict(
        ln_likelihood = ln_lkl,
        variance = variance,
        pe_variance = pe_variance,
        vt_variance = vt_variance,
        num_exp = num_exp,
    )



def ln_likelihood(
    likelihood_ingredients, maximum_variance,
    posteriors, injections, density, parameters,
):
    ingredients = likelihood_ingredients(posteriors, injections, density, parameters)
    ln_lkl = jnp.nan_to_num(ingredients['ln_likelihood'], nan = -jnp.inf, posinf = -jnp.inf)
    variance = jnp.nan_to_num(ingredients['variance'], nan = jnp.inf)
    return jnp.where(variance < maximum_variance, ln_lkl, -jnp.inf)


class BilbyLikelihood(bilby.Likelihood):
    def __init__(self, likelihood_ingredients, posteriors, injections, density, maximum_variance):
        super().__init__({})
        self.num_obs = posteriors['prior'].shape[0]
        self.posteriors = posteriors
        self.injections = injections
        self.maximum_variance = maximum_variance

        self.ln_likelihood = jax.jit(
            lambda posteriors, injections, parameters: ln_likelihood(
                likelihood_ingredients, maximum_variance,
                posteriors, injections, density, parameters,
            ),
        )

        self._likelihood_ingredients = jax.jit(
            lambda posteriors, injections, parameters: likelihood_ingredients(
                posteriors, injections, density, parameters,
            ),
        )

    def log_likelihood(self):
        return self.ln_likelihood(self.posteriors, self.injections, self.parameters)

    def likelihood_ingredients(self, parameters):
        return self._likelihood_ingredients(self.posteriors, self.injections, parameters)


def postprocess(result, likelihood):
    n = len(result.posterior)
    posterior = {k: jnp.array(v) for k, v in result.posterior.items()}

    @jax_tqdm.scan_tqdm(n, print_rate = 1, tqdm_type = 'std')
    def single(carry, x):
        i, parameters = x
        return carry, likelihood.likelihood_ingredients(parameters)

    ingredients = jax.lax.scan(single, None, (jnp.arange(n), posterior))[1]

    # this assume the redshift evolution = 1 at redshift = 0
    if 'ln_vt' in ingredients:
        ingredients['rate_0'] = jax.random.gamma(
            jax.random.key(0), likelihood.num_obs, shape = (n,),
        ) / jnp.exp(ingredients['ln_vt'])

    for k in ingredients:
        result.posterior[k] = np.array(ingredients[k])
    result.save_to_file(overwrite = True, extension = 'hdf5')

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


def evidence(result, likelihood, priors):
    fraction, fraction_error = prior_fraction(likelihood, priors)
    fraction = float(fraction)
    fraction_error = float(fraction_error)
    return dict(
        prior_fraction = fraction,
        prior_fraction_error = fraction_error,
        ln_evidence_bilby = result.log_evidence,
        ln_evidence_bilby_error = result.log_evidence_err,
        ln_evidence = result.log_evidence + np.log(fraction),
        ln_evidence_error = (
            result.log_evidence_err ** 2 + (fraction_error / fraction) ** 2
        ) ** 0.5,
    )
    file = f'{result.outdir}/{result.label}_evidences.json'
    with open(file, 'w') as f:
        json.dump(d, f)
    return d
