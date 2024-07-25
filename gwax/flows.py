import jax
import jax.numpy as jnp
import equinox

from flowjax.bijections import (
    AbstractBijection,
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Invert,
    Stack,
    Tanh,
)
from flowjax.distributions import StandardNormal, Transformed
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.wrappers import non_trainable

from collections.abc import Callable


def Affine(loc = 0, scale = 1):
    affine = AffinePositiveScale(loc, scale)
    loc, scale = jnp.broadcast_arrays(
        affine.loc, jnp.asarray(scale, dtype = float),
    )
    affine = equinox.tree_at(lambda tree: tree.scale, affine, scale)
    return affine


def Logistic(shape = ()):
    loc = jnp.ones(shape) * 0.5
    scale = jnp.ones(shape) * 0.5
    return Chain([Tanh(shape), Affine(loc, scale)])


def UnivariateBounder(bounds = None):
    # no bounds
    if (bounds is None) or all(bound is None for bound in bounds):
        return Identity()

    # bounded on one side
    elif any(bound is None for bound in bounds):
        # bounded on right-hand side
        if bounds[0] is None:
            loc = bounds[1]
            scale = -1
        # bounded on left-hand side
        elif bounds[1] is None:
            loc = bounds[0]
            scale = 1
        return Chain([Exp(), Affine(loc, scale)])

    # bounded on both sides
    else:
        loc = bounds[0]
        scale = bounds[1] - bounds[0]
        return Chain([Logistic(), Affine(loc, scale)])


def Bounder(bounds):
    return Stack(list(map(UnivariateBounder, bounds)))


def Colour(norms):
    mean = jnp.mean(norms, axis = 0)
    std = jnp.std(norms, axis = 0)
    return Affine(loc = mean, scale = std)


def Whiten(norms):
    return Invert(Colour)


def ColourAndBound(bounds = None, norms = None):
    if bounds is None and norms is None:
        return Identity()
    elif bounds is not None and norms is None:
        return Bounder(bounds)
    elif bounds is None and norms is not None:
        return Colour(norms)
    else:
        bounder = Bounder(bounds)
        colour = Colour(jax.vmap(bounder.inverse)(norms))
        return Chain([colour, bounder])


## TODO: interpolation with well-defined derivatives
## monotonic PCHIP
## TODO: add probit after CDF transform
class UnivariateEmpirical(AbstractBijection):
    shape: tuple
    cond_shape: None = None
    _transform: Callable
    _inverse: Callable

    def __init__(self, samples, bounds = None):
        assert len(samples.shape) == 1
        self.shape = ()

        if bounds is None:
            bounds = -jnp.inf, jnp.inf
        else:
            assert len(bounds) == 2
            left = -jnp.inf if bounds[0] is None else bounds[0]
            right = jnp.inf if bounds[1] is None else bounds[1]
            bounds = left, right
        bounds = jnp.nan_to_num(jnp.array(bounds))
        
        points = jnp.sort(jnp.append(samples, bounds))
        cdf = jnp.linspace(0, 1, points.size)
        self._transform = lambda x: jnp.interp(x, points, cdf)
        self._inverse = lambda y: jnp.interp(y, cdf, points)

    def transform(self, x, condition = None):
        return self._transform(x)

    def transform_and_log_det(self, x, condition = None):
        y = self.transform(x)
        grad = jax.grad(self._transform)(x)
        return y, jnp.log(jnp.abs(grad))

    def inverse(self, y, condition = None):
        return self._inverse(y)

    def inverse_and_log_det(self, y, condition = None):
        x = self.inverse(y)
        grad = jax.grad(self._inverse)(y)
        return x, jnp.log(jnp.abs(grad))


def Empirical(samples, bounds):
    return Stack(list(map(UnivariateEmpirical, jnp.asarray(samples).T, bounds)))


class Empirical(AbstractBijection):
    shape: tuple
    cond_shape: None = None
    _bijections: tuple[UnivariateEmpirical]

    def __init__(self, samples, bounds):
        samples = jnp.asarray(samples)
        assert len(samples.shape) == 2
        self.shape = samples.shape[1]
        assert len(bounds) == self.shape
        self._bijections = tuple(map(UnivariateEmpirical, samples.T, bounds))

    def transform(self, x, condition = None):
        single = lambda bijection, x: bijection.transform(x)
        ys = list(map(single, self._bijections, x.T))
        return jnp.stack(ys, axis = -1)

    def transform_and_log_det(self, x, condition = None):
        single = lambda bijection, x: bijection.transform_and_log_det(x)
        ys, log_dets = zip(*map(single, self._bijections, x.T))
        return jnp.stack(ys, axis = -1), sum(log_dets)

    def inverse(self, y, condition = None):
        single = lambda bijection, y: bijection.inverse(y)
        xs = list(map(single, self._bijections, y.T))
        return jnp.stack(xs, axis = -1)

    def inverse_and_log_det(self, y, condition = None):
        single = lambda bijection, y: bijection.inverse_and_log_det(y)
        xs, log_dets = zip(*map(single, self._bijections, y.T))
        return jnp.stack(xs, axis = -1), sum(log_dets)


def bound_from_unbound(flow, bounds = None):
    bounder = Bounder(bounds)

    if type(bounder) is Identity:
        return flow

    base_dist = non_trainable(flow.base_dist)
    bijection = Chain([flow.bijection, non_trainable(bounder)])
    flow = Transformed(base_dist, bijection)

    return flow


def default_flow(key, bounds):
    flow = block_neural_autoregressive_flow(
        key = key,
        base_dist = StandardNormal(shape = (len(bounds),)),
        invert = False,
        nn_depth = 1,
        nn_block_dim = 8,
        flow_layers = 1,
    )
    return bound_from_unbound(flow, bounds)

