import jax.numpy as jnp
from flowjax.bijections import (
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Invert,
    Stack,
    Tanh,
)
from flowjax.utils import arraylike_to_array


def Affine(loc = 0, scale = 1):
    affine = AffinePositiveScale(loc, scale)
    loc, scale = jnp.broadcast_arrays(
        affine.loc, arraylike_to_array(scale, dtype = float),
    )
    affine = equinox.tree_at(lambda tree: tree.scale, affine, scale)
    return affine


def Logistic(shape = ()):
    loc = jnp.ones(shape) * 0.5
    scale = jnp.ones(shape) * 0.5
    return Chain([Tanh(shape), Affine(loc, scale)])


def Normalizer(samples):
    mean = jnp.mean(samples, axis = 0)
    std = jnp.std(samples, axis = 0)
    return Affine(loc = -mean / std, scale = 1 / std)


def Bounder1D(bounds = None):
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
    return Stack(list(map(Bounder1D, bounds)))


def Post(bounds = None, samples = None):
    if bounds is None and norms is None:
        return Identity()
    elif bounds is None and norms is not None:
        return Invert(Normalizer(samples))
    elif bounds is not None and norms is None:
        return Bounder(bounds)
    else:
        bounder = Bounder(bounds)
        normalizer = Normer(jax.vmap(bounder.inverse)(samples))
        return Chain([Invert(normalizer), bounder])

