import bilby
import numpy as np
import wcosmo; wcosmo.disable_units()


def detector_to_source_jacobian(redshift, H0, Om0):
    return wcosmo.dDLdz(redshift, H0, Om0) * (1 + redshift) ** 2

def detector_to_source(mass_1, mass_2, luminosity_distance, H0, Om0):
    redshift = wcosmo.z_at_value(
        wcosmo.luminosity_distance, luminosity_distance, H0 = H0, Om0 = Om0,
    )
    mass_1_source = mass_1 / (1 + redshift)
    mass_2_source = mass_2 / (1 + redshift)
    jacobian = detector_to_source_jacobian(redshift, H0, Om0)
    return mass_1_source, mass_2_source, redshift, jacobian

def source_to_detector(mass_1_source, mass_2_source, redshift, H0, Om0):
    luminosity_distance = wcosmo.luminosity_distance(redshift, H0, Om0)
    mass_1 = mass_1_source * (1 + redshift)
    mass_2 = mass_2_source * (1 + redshift)
    jacobian = detector_to_source_jacobian(redshift, H0, Om0)
    return mass_1, mass_2, luminosity_distance, 1 / jacobian

def data_to_source(data, H0, Om0):
    mass_1_source, mass_2_source, redshift, jacobian = detector_to_source(
        data['mass_1'], data['mass_2'], data['luminosity_distance'], H0, Om0,
    )
    data['mass_1_source'] = mass_1_source
    data['mass_2_source'] = mass_2_source
    data['redshift'] = redshift
    data['weight'] /= jacobian
    data.pop('mass_1')
    data.pop('mass_2')
    data.pop('luminosity_distance')
    return data

def data_to_detector(data, H0, Om0):
    mass_1, mass_2, luminosity_distance, jacobian = source_to_detector(
        data['mass_1_source'], data['mass_2_source'], data['reshuft'], H0, Om0,
    )
    data['mass_1'] = mass_1
    data['mass_2'] = mass_2
    data['luminosity_distance'] = luminosity_distance
    data['weight'] /= jacobian
    data.pop('mass_1_source')
    data.pop('mass_2_source')
    data.pop('redshift')
    return data

def comoving_to_detector(redshift, H0, Om0):
    return (
        wcosmo.differential_comoving_volume(redshift, H0, Om0)
        * 4 * np.pi / 1e9 / (1 + redshift)
    )
