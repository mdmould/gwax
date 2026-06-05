import numpy as np
import wcosmo; wcosmo.disable_units()

from gwax.cosmology import detector_to_source, comoving_to_detector


def demagnify(
    mass_1, mass_2, luminosity_distance, magnification, H0 = 67.9, Om0 = 0.3065,
):
    mass_1_source, mass_2_source, redshift, jacobian = detector_to_source(
        mass_1, mass_2, luminosity_distance * magnification ** 0.5, H0, Om0,
    )
    weight = (
        comoving_to_detector(redshift, H0, Om0)
        * magnification ** 0.5
        / jacobian
    )
    return mass_1_source, mass_2_source, redshift, weight


def lensing_prior(redshift, prefactor, H0 = 67.9, Om0 = 0.3065):
    dc = wcosmo.comoving_distance(redshift, H0, Om0)
    dh = wcosmo.hubble_distance(H0)
    return prefactor * (dc / dh) ** 3


def sample_magnification(shape):
    slope = -3
    low = 2
    u = np.random.random(shape)
    mu = low * (1 - u) ** (1 / (slope + 1))
    return mu


def demagnify_data(data, magnification):
    (
        data['mass_1_source'],
        data['mass_2_source'],
        data['redshift'],
        weight,
    ) = demagnify(
        data.pop('mass_1'),
        data.pop('mass_2'),
        data.pop('luminosity_distance'),
        magnification,
    )
    data['weight'] = data['weight'] * weight
    return data


def convert_not_lensed(data, prefactor):
    magnification = 1
    data = demagnify_data(data, magnification)
    data['weight'] *= 1 - lensing_prior(data['redshift'], prefactor)
    return data

def convert_lensed(data, prefactor):
    magnification = sample_magnification(data['weight'].shape)
    data = demagnify_data(data, magnification)
    data['weight'] *= lensing_prior(data['redshift'], prefactor)
    return data

def convert_marginalized(data, prefactor):
    old_data = convert_not_lensed(data.copy(), prefactor)
    new_data = convert_lensed(data.copy(), prefactor)
    data = {
        key: np.concatenate([old_data[key], new_data[key]], axis = -1)
        for key in new_data if key not in ('total', 'time')
    }
    for key in new_data:
        if key not in data:
            data[key] = new_data[key]
    return data
