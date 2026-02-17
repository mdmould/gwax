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

def comoving_to_detector(redshift, H0, Om0):
    return (
        wcosmo.differential_comoving_volume(redshift, H0, Om0)
        * 4 * np.pi / 1e9 / (1 + redshift)
    )

def comoving_to_detector_lal(redshift):
    cosmo = bilby.gw.cosmology.get_cosmology('Planck15_LAL')
    H0 = float(cosmo.H0.value)
    Om0 = float(cosmo.Om0)
    return comoving_to_detector(redshift, Om0, H0)
