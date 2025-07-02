import jax
import jax.numpy as jnp


def cubic_filter(x):
    return (3 - 2 * x) * x**2 * (0 <= x) * (x <= 1) + (1 < x)

def highpass(x, xmin, dmin):
    return cubic_filter((x - xmin) / dmin)

def lowpass(x, xmax, dmax):
    return highpass(x, xmax, -dmax)

def bandpass(x, xmin, xmax, dmin, dmax):
    return highpass(x, xmin, dmin) * lowpass(x, xmax, dmax)


def truncated_powerlaw(x, alpha, xmin, xmax):
    cut = (xmin <= x) * (x <= xmax)
    shape = x**alpha
    norm = (xmax**(alpha + 1) - xmin**(alpha + 1)) / (alpha + 1)
    return (shape / norm) * cut

def powerlaw_integral(x, alpha, loc, delta):
    a, c, d = alpha, loc, delta
    return (
        3 * (2 * c + (4 + a) * d)
        * (c**2 / (1 + a) - 2 * c * x / (2 + a) + x**2 / (3 + a))
        - 2 * (x - c)**3
    ) * x**(1 + a) / (4 + a) / d**3

def highpass_truncated_powerlaw(x, alpha, xmin, xmax, dmin):
    cut = (xmin <= x) * (x <= xmax)
    shape = x**alpha * highpass(x, xmin, dmin)
    norm = (
        - powerlaw_integral(xmin, alpha, xmin, dmin)
        + powerlaw_integral(xmax + dmin, alpha, xmin, dmin)
        - (xmin + dmin) ** (alpha + 1) / (alpha + 1)
        + xmax ** (alpha + 1) / (alpha + 1)
    )
    return (shape / norm) * cut

def highpass_powerlaw(x, alpha, xmin, dmin):
    cut = xmin <= x
    shape = x**alpha * highpass(x, xmin, dmin)
    norm = (
        - powerlaw_integral(xmin, alpha, xmin, dmin)
        + powerlaw_integral(xmin + dmin, alpha, xmin, dmin)
        - (xmin + dmin) ** (alpha + 1) / (alpha + 1)
    )
    return (shape / norm) * cut

def bandpass_powerlaw(x, alpha, xmin, xmax, dmin, dmax):
    cut = (xmin <= x) * (x <= xmax)
    shape = x**alpha * bandpass(x, xmin, xmax, dmin, dmax)
    norm = (
        - powerlaw_integral(xmin, alpha, xmin, dmin)
        + powerlaw_integral(xmin + dmin, alpha, xmin, dmin)
        - (xmin + dmin) ** (alpha + 1) / (alpha + 1)
        + (xmax - dmax) ** (alpha + 1) / (alpha + 1)
        - powerlaw_integral(xmax - dmax, alpha, xmax, -dmax)
        + powerlaw_integral(xmax, alpha, xmax, -dmax)
    )
    return (shape / norm) * cut


def truncated_normal(x, mu, sigma, xmin, xmax):
    cut = (xmin <= x) * (x <= xmax)
    shape = jax.scipy.stats.norm.pdf(x, mu, sigma)
    norm = (
        - jax.scipy.stats.norm.cdf(xmin, mu, sigma)
        + jax.scipy.stats.norm.cdf(xmax, mu, sigma)
    )
    return (shape / norm) * cut

def normal_integral(x, mu, sigma, loc, delta):
    m, s, c, d = mu, sigma, loc, delta
    return (
        jnp.exp(-(x - m)**2 / 2 / s ** 2) * (2 / jnp.pi)**0.5 * s * (
            6 * c * (c + d - m - x)
            - 3 * d * (m + x)
            + 2 * (m**2 + 2 * s**2 + m * x + x**2)
        )
        - jax.lax.erf((m - x) / s / 2**0.5) * (
            (2 * c + 3 * d - 2 * m) * (c - m)**2
            + 3 * s**2 * (2 * c + d - 2 * m)
        )
    ) / 2 / d**3

def highpass_normal(x, mu, sigma, xmin, dmin):
    cut = xmin <= x
    shape = jax.scipy.stats.norm.pdf(x, mu, sigma) * highpass(x, xmin, dmin)
    norm = (
        - normal_integral(xmin, mu, sigma, xmin, dmin)
        + normal_integral(xmin + dmin, mu, sigma, xmin, dmin)
        - jax.scipy.stats.norm.cdf(xmin + dmin, mu, sigma)
        + 1
    )
    return (shape / norm) * cut

def highpass_truncated_normal(x, mu, sigma, xmin, xmax, dmin):
    cut = (xmin <= x) * (x <= xmax)
    shape = jax.scipy.stats.norm.pdf(x, mu, sigma) * highpass(x, xmin, dmin)
    norm = (
        - normal_integral(xmin, mu, sigma, xmin, dmin)
        + normal_integral(xmin + dmin, mu, sigma, xmin, dmin)
        - jax.scipy.stats.norm.cdf(xmin + dmin, mu, sigma)
        + jax.scipy.stats.norm.cdf(xmax, mu, sigma)
    )
    return (shape / norm) * cut

def bandpass_normal(x, mu, sigma, xmin, xmax, dmin, dmax):
    cut = (xmin <= x) * (x <= xmax)
    shape = (
        jax.scipy.stats.norm.pdf(x, mu, sigma)
        * bandpass(x, xmin, xmax, dmin, dmax)
    )
    norm = (
        - normal_integral(xmin, mu, sigma, xmin, dmin)
        + normal_integral(xmin + dmin, mu, sigma, xmin, dmin)
        - jax.scipy.stats.norm.cdf(xmin + dmin, mu, sigma)
        + jax.scipy.stats.norm.cdf(xmax - dmax, mu, sigma)
        - normal_integral(xmax - dmax, mu, sigma, xmax, -dmax)
        + normal_integral(xmax, mu, sigma, xmax, -dmax)
    )
    return (shape / norm) * cut
