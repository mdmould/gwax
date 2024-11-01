import jax
import jax.numpy as jnp
import equinox

from flowjax.bijections import (
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Stack,
    Tanh,
)
from flowjax.distributions import StandardNormal, Transformed
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.wrappers import non_trainable


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


def bound_from_unbound(flow, bounds = None):
    bounder = Bounder(bounds)

    if all(type(b) is Identity for b in bounder.bijections):
        bijection = flow.bijection
    else:
        bijection = Chain([flow.bijection, non_trainable(bounder)])

    return Transformed(non_trainable(flow.base_dist), bijection)


def default_flow(key, bounds, **kwargs):
    default_kwargs = dict(
        key = key,
        base_dist = StandardNormal(shape = (len(bounds),)),
        invert = False,
        nn_depth = 1,
        nn_block_dim = 8,
        flow_layers = 1,
    )
    
    for arg in kwargs:
        default_kwargs[arg] = kwargs[arg]

    flow = block_neural_autoregressive_flow(**default_kwargs)

    return bound_from_unbound(flow, bounds)
