import bilby
import numpy as np
import wcosmo; wcosmo.disable_units()


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
