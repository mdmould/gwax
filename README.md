# gwax

Gravitational-wave astronomy in JAX.

## Install

1. Install JAX following their [instructions](https://github.com/google/jax#installation) for your platform and hardware.
2. `pip install gwax`.

To install the latest development: `pip install git+https://github.com/mdmould/gwax`.

## Implementations

- Flow-based variational inference for gravitational-wave population analysis.

## Usage

See the [examples](https://github.com/mdmould/gwax/tree/main/examples).

## Current limitations

- gwax is being actively developed and the interface is likely to undergo breaking changes.
- Currently the only thing implemented is variation inference of gravitational-wave populations.
- Variational population inference only interfaces with [gwpopulation](https://github.com/ColmTalbot/gwpopulation).
- Only univariate uniform priors are supported.
