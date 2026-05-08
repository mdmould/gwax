# gwax

Gravitational-wave astronomy in JAX.

## Install

1. Install JAX following their [instructions](https://github.com/google/jax#installation) for your platform and hardware.
2. `pip install gwax`.

To install the latest development: `pip install git+https://github.com/mdmould/gwax`.

## Implementations

- Helper functions to load LVK data.
- Likelihood for gravitational-wave population analysis, including a new implementation that avoids single-event downsampling.
- Some standard parametric population models.
- Nonparametric population models.
- Flow-based variational inference, e.g., for population analyses.

## Usage

See the [examples](https://github.com/mdmould/gwax/tree/main/examples).

## Limitations

Note that I made this repo primarily to contain code I reuse many times, not as a comprehensive and documented package. Please get in touch if you'd like help using it 👍
