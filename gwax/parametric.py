import jax
import jax.numpy as jnp

import wcosmo
wcosmo.disable_units()


def cubic_filter(x):
    return (3 - 2 * x) * x ** 2 * (0 <= x) * (x <= 1) + (1 < x)

def highpass(x, loc, delta):
    return cubic_filter((x - loc) / delta)

def lowpass(x, loc, delta):
    return highpass(x, loc, - delta)

def bandpass(x, lo, hi, dlo, dhi):
    return highpass(x, lo, dlo) * lowpass(x, hi, dhi)


def truncated_normal(x, mu, sigma, lo, hi):
    cut = (lo <= x) * (x <= hi)
    shape = jax.scipy.stats.norm.pdf(x, mu, sigma)
    norm = (
        - jax.scipy.stats.norm.cdf(lo, mu, sigma)
        + jax.scipy.stats.norm.cdf(hi, mu, sigma)
    )
    return cut * shape / norm

def normal_integral(x, mu, sigma, loc, delta):
    m, s, c, d = mu, sigma, loc, delta
    return (
        jnp.exp(-(x - m) ** 2 / 2 / s ** 2) * (2 / jnp.pi) ** 0.5 * s * (
            6 * c * (c + d - m - x)
            - 3 * d * (m + x)
            + 2 * (m ** 2 + 2 * s**2 + m * x + x ** 2)
        )
        - jax.lax.erf((m - x) / s / 2 ** 0.5) * (
            (2 * c + 3 * d - 2 * m) * (c - m) ** 2
            + 3 * s ** 2 * (2 * c + d - 2 * m)
        )
    ) / 2 / d ** 3

# def highpass_truncated_normal(x, mu, sigma, lo, hi, dlo):
#     cut = (lo <= x) * (x <= hi)
#     shape = jax.scipy.stats.norm.pdf(x, mu, sigma) * highpass(x, lo, dlo)
#     norm = (
#         - normal_integral(lo, mu, sigma, lo, dlo)
#         + normal_integral(lo + dlo, mu, sigma, lo, dlo)
#     ) + (
#         - jax.scipy.stats.norm.cdf(lo + dlo, mu, sigma)
#         + jax.scipy.stats.norm.cdf(hi, mu, sigma)
#     )
#     return cut * shape / norm

def highpass_truncated_normal(x, mu, sigma, lo, hi, dlo):
    cut = (lo <= x) * (x <= hi)
    shape = jax.scipy.stats.norm.pdf(x, mu, sigma) * highpass(x, lo, dlo)
    x0 = jnp.minimum(lo + dlo, hi)
    norm = (
        - normal_integral(lo, mu, sigma, lo, dlo)
        + normal_integral(x0, mu, sigma, lo, dlo)
    ) + (
        - jax.scipy.stats.norm.cdf(x0, mu, sigma)
        + jax.scipy.stats.norm.cdf(hi, mu, sigma)
    )
    return cut * shape / norm

def bandpass_normal(x, mu, sigma, lo, hi, dlo, dhi):
    cut = (lo <= x) * (x <= hi)
    shape = jax.scipy.stats.norm.pdf(x, mu, sigma) * bandpass(x, lo, hi, dlo, dhi)
    norm = (
        - normal_integral(lo, mu, sigma, lo, dlo)
        + normal_integral(lo + dlo, mu, sigma, lo, dlo)
    ) + (
        - jax.scipy.stats.norm.cdf(lo + dlo, mu, sigma)
        + jax.scipy.stats.norm.cdf(hi - dhi, mu, sigma)
    ) + (
        - normal_integral(hi - dhi, mu, sigma, hi, - dhi)
        + normal_integral(hi, mu, sigma, hi, - dhi)
    )
    return cut * shape / norm


def truncated_powerlaw(x, alpha, lo, hi):
    cut = (lo <= x) * (x <= hi)
    shape = x ** alpha
    norm = (hi ** (alpha + 1) - lo ** (alpha + 1)) / (alpha + 1)
    return cut * shape / norm

def powerlaw_integral(x, alpha, loc, delta):
    a, c, d = alpha, loc, delta
    return (
        3 * (2 * c + (4 + a) * d)
        * (c ** 2 / (1 + a) - 2 * c * x / (2 + a) + x ** 2 / (3 + a))
        - 2 * (x - c) ** 3
    ) * x ** (1 + a) / (4 + a) / d ** 3

# def highpass_truncated_powerlaw(x, alpha, lo, hi, dlo):
#     cut = (lo <= x) * (x <= hi)
#     shape = x ** alpha * highpass(x, lo, dlo)
#     norm = (
#         - powerlaw_integral(lo, alpha, lo, dlo)
#         + powerlaw_integral(lo + dlo, alpha, lo, dlo)
#     ) + (
#         - (lo + dlo) ** (alpha + 1) / (alpha + 1)
#         + hi ** (alpha + 1) / (alpha + 1)
#     )
#     return cut * shape / norm

def highpass_truncated_powerlaw(x, alpha, lo, hi, dlo):
    cut = (lo <= x) * (x <= hi)
    shape = x ** alpha * highpass(x, lo, dlo)
    x0 = jnp.minimum(lo + dlo, hi)
    norm = (
        - powerlaw_integral(lo, alpha, lo, dlo)
        + powerlaw_integral(x0, alpha, lo, dlo)
    ) + (
        - x0 ** (alpha + 1) / (alpha + 1)
        + hi ** (alpha + 1) / (alpha + 1)
    )
    return cut * shape / norm

def bandpass_powerlaw(x, alpha, lo, hi, dlo, dhi):
    cut = (lo <= x) * (x <= hi)
    shape = x ** alpha * bandpass(x, lo, hi, dlo, dhi)
    norm = (
        - powerlaw_integral(lo, alpha, lo, dlo)
        + powerlaw_integral(lo + dlo, alpha, lo, dlo)
    ) + (
        - (lo + dlo) ** (alpha + 1) / (alpha + 1)
        + (hi - dhi) ** (alpha + 1) / (alpha + 1)
    ) + (
        - powerlaw_integral(hi - dhi, alpha, hi, - dhi)
        + powerlaw_integral(hi, alpha, hi, - dhi)
    )
    return cut * shape / norm

def truncated_broken_powerlaw(x, alpha1, alpha2, loc, lo, hi):
    cut = (lo <= x) * (x <= hi)
    shape = (x / loc) ** jnp.where(x <= loc, alpha1, alpha2)
    norm = (
        - lo ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
        + loc ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
    ) + (
        - loc ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
        + hi ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
    )
    return cut * shape / norm

# def highpass_truncated_broken_powerlaw(x, alpha1, alpha2, loc, lo, hi, dlo):
#     cut = (lo <= x) * (x <= hi)
#     alpha = jnp.where(x <= loc, alpha1, alpha2)
#     shape = (x / loc) ** alpha * highpass(x, lo, dlo)
#     norm = (
#         - powerlaw_integral(lo, alpha1, lo, dlo) / loc ** alpha1
#         + powerlaw_integral(lo + dlo, alpha1, lo, dlo) / loc ** alpha1
#     ) + (
#         - (lo + dlo) ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
#         + loc ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
#     ) + (
#         - loc ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
#         + hi ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
#     )
#     return cut * shape / norm

def highpass_truncated_broken_powerlaw(x, alpha1, alpha2, loc, lo, hi, dlo):
    cut = (lo <= x) * (x <= hi)
    alpha = jnp.where(x <= loc, alpha1, alpha2)
    shape = (x / loc) ** alpha * highpass(x, lo, dlo)
    x0 = jnp.minimum(lo + dlo, hi)
    x1 = jnp.minimum(loc, hi)
    norm = (
        - powerlaw_integral(lo, alpha1, lo, dlo) / loc ** alpha1
        + powerlaw_integral(x0, alpha1, lo, dlo) / loc ** alpha1
    ) + (
        - x0 ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
        + x1 ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
    ) + (
        - x1 ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
        + hi ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
    )
    return cut * shape / norm

def bandpass_broken_powerlaw(x, alpha1, alpha2, loc, lo, hi, dlo, dhi):
    cut = (lo <= x) * (x <= hi)
    shape = (x / loc) ** jnp.where(x <= loc, alpha1, alpha2) * bandpass(x, lo, hi, dlo, dhi)
    norm = (
        - powerlaw_integral(lo, alpha1, lo, dlo) / loc ** alpha1
        + powerlaw_integral(lo + dlo, alpha1, lo, dlo) / loc ** alpha1
    ) + (
        - (lo + dlo) ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
        + loc ** (alpha1 + 1) / (alpha1 + 1) / loc ** alpha1
    ) + (
        - loc ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
        + (hi - dhi) ** (alpha2 + 1) / (alpha2 + 1) / loc ** alpha2
    ) + (
        - powerlaw_integral(hi - dhi, alpha2, hi, -dhi) / loc ** alpha2
        + powerlaw_integral(hi, alpha2, hi, -dhi) / loc ** alpha2
    )
    return cut * shape / norm
