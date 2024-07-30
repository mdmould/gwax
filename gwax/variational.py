import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax

from flowjax.distributions import Uniform
from flowjax.wrappers import NonTrainable

from .flows import default_flow


def get_prior(bounds):
    lo = jnp.array(bounds)[:, 0]
    hi = jnp.array(bounds)[:, 1]
    return Uniform(minval = lo, maxval = hi)


def get_log_likelihood(likelihood = None, return_variance = False):
    if likelihood is None:
        if return_variance:
            return lambda parameters: (0.0, 0.0)
        return lambda parameters: 0.0

    if return_variance:
        def log_likelihood_and_variance(parameters):
            likelihood.parameters.update(parameters)
            return likelihood.ln_likelihood_and_variance()

        return log_likelihood_and_variance

    def log_likelihood(parameters):
        likelihood.parameters.update(parameters)
        return likelihood.log_likelihood_ratio()

    return log_likelihood


def likelihood_extras(likelihood, parameters):
    likelihood.parameters.update(parameters)
    likelihood.parameters, added_keys = likelihood.conversion_function(
        likelihood.parameters,
    )
    likelihood.hyper_prior.parameters.update(parameters)

    log_bayes_factors, variances = \
        likelihood._compute_per_event_ln_bayes_factors()

    detection_efficiency, detection_variance = \
        likelihood.selection_function.detection_efficiency(parameters)

    selection = - likelihood.n_posteriors * jnp.log(detection_efficiency)
    selection_variance = (
        likelihood.n_posteriors ** 2
        * detection_variance
        / detection_efficiency ** 2
    )

    log_likelihood = jnp.sum(log_bayes_factors) + selection
    variance = jnp.sum(variances) + selection_variance

    return dict(
        log_likelihood = log_likelihood,
        variance = variance,
        log_bayes_factors = log_bayes_factors,
        variances = variances,
        detection_efficiency = detection_efficiency,
        detection_variance = detection_variance,
        selection = selection,
        selection_variance = selection_variance,
    )


def reverse_loss(key, batch_size, flow, log_target, temper = 1.0):
    samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
    log_targets = log_target(samples) * temper
    return jnp.mean(log_flows - log_targets)


def trainer(
    key,
    prior_bounds,
    likelihood = None,
    flow = None,
    batch_size = 1,
    steps = 1_000,
    learning_rate = 1e-2,
    optimizer = None,
    taper = None,
    temper_schedule = None,
    tqdm_args = {},
):
    names = tuple(prior_bounds.keys())
    bounds = tuple(prior_bounds.values())
    prior = get_prior(bounds)
    log_likelihood_and_variance = get_log_likelihood(likelihood, True)

    if taper is None:
        taper = lambda variance: 0.0

    def log_target(samples):
        log_priors = prior.log_prob(samples)
        parameters = dict(zip(names, samples.T))
        log_lkls, variances = jax.vmap(log_likelihood_and_variance)(parameters)
        # log_lkls, variances = jax.lax.map(log_likelihood_and_variance, parameters)
        return log_priors + log_lkls + taper(variances)

    if flow is None:
        key, _key = jax.random.split(key)
        flow = default_flow(_key, prior_bounds.values())

    params, static = equinox.partition(
        pytree = flow,
        filter_spec = equinox.is_inexact_array,
        is_leaf = lambda leaf: isinstance(leaf, NonTrainable),
    )

    if temper_schedule is None:
        temper_schedule = lambda step: 1.0

    @equinox.filter_value_and_grad
    def loss_and_grad(params, key, step):
        flow = equinox.combine(params, static)
        temper = temper_schedule(step)
        return reverse_loss(key, batch_size, flow, log_target, temper)

    if optimizer is None:
        optimizer = optax.adam
    if callable(optimizer):
        optimizer = optimizer(learning_rate)

    state = optimizer.init(params)

    tqdm_defaults = dict(print_rate = 1, tqdm_type = 'auto')
    for arg in tqdm_args:
        tqdm_defaults[arg] = tqdm_args[arg]    

    @jax_tqdm.scan_tqdm(steps, **tqdm_defaults)
    @equinox.filter_jit
    def update(carry, step):
        key, params, state = carry
        key, _key = jax.random.split(key)
        loss, grad = loss_and_grad(params, _key, step)
        updates, state = optimizer.update(grad, state, params)
        params = equinox.apply_updates(params, updates)
        return (key, params, state), loss

    (key, params, state), losses = jax.lax.scan(
        update, (key, params, state), jnp.arange(steps),
    )
    flow = equinox.combine(params, static)

    return flow, losses


def importance(key, prior_bounds, likelihood, flow, batch_size):
    names = tuple(prior_bounds.keys())
    bounds = tuple(prior_bounds.values())
    prior = get_prior(bounds)
    log_likelihood = equinox.filter_jit(get_log_likelihood(likelihood, False))
    
    samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
    log_priors = prior.log_prob(samples)
    parameters = dict(zip(names, samples.T))
    # log_lkls = jax.vmap(log_likelihood)(parameters)
    log_lkls = jax.lax.map(log_likelihood, parameters)
    log_weights = log_priors + log_lkls - log_flows

    log_evidence = jax.nn.logsumexp(log_weights) - jnp.log(batch_size)

    log_sq_mean = 2 * log_evidence
    log_mean_sq = jax.nn.logsumexp(2 * log_weights) - jnp.log(batch_size)
    
    var_evidence = (jnp.exp(log_mean_sq) - jnp.exp(log_sq_mean)) / batch_size
    var_log_evidence = var_evidence / jnp.exp(2 * log_evidence)

    log_neff = jnp.log(batch_size) + log_sq_mean - log_mean_sq
    
    return dict(
        samples = samples,
        weights = jnp.exp(log_weights),
        efficiency = jnp.exp(log_neff) / batch_size,
        log_evidence = log_evidence,
        log_evidence_variance = var_log_evidence,
        log_likelihoods = log_lkls,
    )
