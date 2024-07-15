from functools import partial
import jax
import jax.numpy as jnp
import equinox

from flowjax.bijections import (
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Invert,
    Stack,
    Tanh,
)


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


def _Bounder(bounds = None):
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
        return Chain([Logisitic(), Affine(loc, scale)])


def Bounder(bounds):
    return Stack(list(map(_Bounder, bounds)))


def bound_from_unbound(flow, bounds = None):
    bounder = Bounder(bounds)

    if type(bounder) is Identity:
        return flow
    
    flow = Transformed(flow.base_dist, Chain([flow.bijection, bounder]))
    flow = equinox.tree_at(
        lambda tree: tree.base_dist, flow, replace_fn = non_trainable,
    )
    flow = equinox.tree_at(
        lambda tree: tree.bijection[-1], flow, replace_fn = non_trainable,
    )

    return flow
