from glob import glob
import os
import time

import bilby
from gwpopulation_pipe.analytic_spin_prior import (
    chi_effective_prior_from_isotropic_spins,
    prior_chieff_chip_isotropic,
)
import h5py
import numpy as np
import tqdm
import wcosmo; wcosmo.disable_units()


def get_events_list(catalog = 'GWTC-4', min_ifar = 1, bbh = True, er = False):
    url = 'https://gwosc.org/eventapi/ascii/query/show?release='
    url += ','.join(
        [
            'GWTC-1-confident,GWTC-1-marginal',
            'GWTC-2.1-confident,GWTC-2.1-marginal',
            'GWTC-3-confident,GWTC-3-marginal',
            'GWTC-4.0',
        ][:int(catalog.split('-')[1][0])]
    )
    url += f'&min-mass-2-source={3 if bbh else 0}&max-far={1 / min_ifar}'
    temp = f'./events-{time.time_ns()}.txt'
    os.system(f'wget -O {temp} "{url}"')
    events = np.loadtxt(temp, dtype = str, skiprows = 1, usecols = 1)
    os.system(f'rm {temp}')
    events = sorted(map(str, np.unique(events)))
    if not er:
        events.remove('GW230518_125908')
    return events

def get_event_file(path, catalog, event):
    if catalog == 'GWTC-2':
        files = glob(f'{path}/lvk-data/{catalog}/PE/{event}.h5')
    elif catalog in ['GWTC-2.1', 'GWTC-3']:
        files = glob(f'{path}/lvk-data/{catalog}/PE/*{event}*_nocosmo.h5')
    else: # GWTC-1, GWTC-4s
        files = glob(f'{path}/lvk-data/{catalog}/PE/*{event}*.hdf5')
    if event == 'GW190521':
        files = [file for file in files if 'GW190521_074359' not in file]
    assert len(files) == 1
    return files[0]

def get_event_catalog_and_file(path, event):
    files = {}
    for catalog in 'GWTC-1', 'GWTC-2', 'GWTC-2.1', 'GWTC-3', 'GWTC-4':
        try:
            files[catalog] = get_event_file(path, catalog, event)
        except:
            pass
    if 'GWTC-1' in files and 'GWTC-2.1' in files:
        catalog = 'GWTC-2.1'
    elif 'GWTC-2' in files and 'GWTC-2.1' in files:
        with h5py.File(files['GWTC-2']) as f:
            if 'C01:NRSur7dq4' in f:
                catalog = 'GWTC-2'
            else:
                catalog = 'GWTC-2.1'
    else:
        assert len(files) == 1
        catalog = list(files)[0]
    return catalog, files[catalog]

def waveform_priority(event, catalog, analyses):
    event_specific = dict(
        GW170817 = ['IMRPhenomPv2NRT_highSpin_posterior'],
        GW190425 = ['C01:IMRPhenomPv2_NRTidal:HighSpin'],
        GW191219_163120 = ['C01:IMRPhenomXPHM:HighSpin', 'C01:SEOBNRv4PHM'],
        GW200115_042309 = ['C01:IMRPhenomXPHM:HighSpin', 'C01:SEOBNRv4PHM'],
        GW230518_125908 = ['C00:IMRPhenomXPHM-SpinTaylor'],
        GW230529_181500 = [
            'C00:IMRPhenomXPHM:HighSpin', 'C00:SEOBNRv5PHM:HighSpin',
        ],
    )
    if event in event_specific:
        return event_specific[event]
    elif catalog == 'GWTC-1':
        return ['IMRPhenomPv2_posterior', 'SEOBNRv3_posterior']
    elif catalog == 'GWTC-2':
        return ['C01:NRSur7dq4']
    elif catalog == 'GWTC-2.1':
        if 'C01:SEOBNRv4PHM' in analyses:
            return ['C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']
        else:
            return ['C01:IMRPhenomXPHM']
    elif catalog == 'GWTC-3':
        return ['C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']
    elif catalog == 'GWTC-4':
        if 'C00:NRSur7dq4' in analyses:
            return ['C00:NRSur7dq4']
        else:
            return ['C00:IMRPhenomXPHM-SpinTaylor', 'C00:SEOBNRv5PHM']

def get_event_samples(file, analyses, keys):
    swap = dict(
        a_1 = 'spin1',
        a_2 = 'spin2',
        cos_tilt_1 = 'costilt1',
        cos_tilt_2 = 'costilt2',
        mass_1 = 'm1_detector_frame_Msun',
        mass_2 = 'm2_detector_frame_Msun',
        luminosity_distance = 'luminosity_distance_Mpc',
        cos_theta_jn = 'costheta_jn',
        ra = 'right_ascension',
        dec = 'declination',
    )
    samples = {}
    with h5py.File(file) as f:
        for analysis in analyses:
            if 'GWTC-1' in file:
                data = f[analysis]
                samples[analysis] = {key: data[swap[key]] for key in keys}
            else:
                data = f[analysis]['posterior_samples']
                samples[analysis] = {key: data[key] for key in keys}
    return samples

def get_event(path, event, keys):
    catalog, file = get_event_catalog_and_file(path, event)
    with h5py.File(file) as f:
        analyses = waveform_priority(event, catalog, f)
    data = get_event_samples(file, analyses, keys)
    data['catalog'] = catalog
    data['file'] = file
    return data

def get_events(path, events, keys):
    data = {}
    for event in tqdm.tqdm(events):
        data[event] = get_event(path, event, keys)
    return data


def eval_chi_eff(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    a_1z = a_1 * cos_tilt_1
    a_2z = a_2 * cos_tilt_2
    return (a_1z + mass_ratio * a_2z) / (1 + mass_ratio)

def eval_chi_p(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    a_1p = a_1 * np.sin(np.arccos(cos_tilt_1))
    a_2p = a_2 * np.sin(np.arccos(cos_tilt_2))
    coeff = mass_ratio * (4 * mass_ratio + 3) / (4 + 3 * mass_ratio)
    return np.maximum(a_1p, a_2p * coeff)


def prior_euclidean(luminosity_distance):
    return luminosity_distance ** 2

def prior_comoving_source(luminosity_distance, H0, Om0):
    redshift = wcosmo.z_at_value(
        wcosmo.luminosity_distance, luminosity_distance, H0 = H0, Om0 = Om0,
    )
    dd_dz = wcosmo.dDLdz(redshift, H0, Om0)
    dv_dz = wcosmo.differential_comoving_volume(redshift, H0, Om0)
    return dv_dz / (1 + redshift) / dd_dz


def get_posteriors(
    path,
    catalog = 'GWTC-4',
    min_ifar = 1,
    bbh = True,
    er = False,
    exclude = [],
    source = True,
    mass_ratio = False,
    chi_eff = False,
    chi_p = False,
    extra_keys = [],
    downsample = False,
    stack = True,
):
    exclude = sorted(set(exclude))
    events = get_events_list(catalog, min_ifar, bbh, er)

    keys = [
        'luminosity_distance',
        'mass_1',
        'mass_2',
        'a_1',
        'a_2',
        'cos_tilt_1',
        'cos_tilt_2',
    ]
    posteriors = {}
    for event in tqdm.tqdm(events):
        if event in exclude:
            print(f'excluding {event}: in exclude list')
            continue
        posteriors[event] = get_event(path, event, keys)



    return


def get_injections(
    path,
    catalog = 'GWTC-4',
    min_ifar = 1,
    min_snr = 10,
    source = True,
    mass_ratio = False,
    chi_eff = False,
    chi_p = False,
    extra_keys = [],
):
    file = f'{path}/lvk-data/GWTC-4/VT/mixture-semi_o1_o2-real_o3'
    if catalog == 'GWTC-3':
        file += '-cartesian_spins_20250503134659UTC.hdf'
    elif catalog == 'GWTC-4':
        file += '_o4a-cartesian_spins_20250503134659UTC.hdf'
    print(file)

    injections = {}

    with h5py.File(file, 'r') as f:
        injections['time'] = \
            f.attrs['total_analysis_time'] / 60 / 60 / 24 / 365.25
        injections['total'] = f.attrs['total_generated']

        d = f['events'][:]

        far = np.min([d[k] for k in d.dtype.names if 'far' in k], axis = 0)
        snr = d['semianalytic_observed_phase_maximized_snr_net']
        found = (far < 1 / min_ifar) | (snr > min_snr)

        prior = np.exp(d[
            'lnpdraw_mass1_source_mass2_source_redshift'
            '_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'
        ][found]) / d['weights'][found]

        z = d['redshift'][found]
        m1 = d['mass1_source'][found]
        m2 = d['mass2_source'][found]
        s1x = d['spin1x'][found]
        s1y = d['spin1y'][found]
        s1z = d['spin1z'][found]
        s2x = d['spin2x'][found]
        s2y = d['spin2y'][found]
        s2z = d['spin2z'][found]

        q = m2 / m1
        a1 = (s1x ** 2 + s1y ** 2 + s1z ** 2) ** 0.5
        a2 = (s2x ** 2 + s2y ** 2 + s2z ** 2) ** 0.5
        c1 = s1z / a1
        c2 = s2z / a2

        prior *= a1 ** 2 * a2 ** 2 # (x, y, z) -> (a, cos(theta), phi)

        injections['redshift'] = z
        injections['mass_1_source'] = m1

        if mass_ratio:
            injections['mass_ratio'] = q
            prior *= m1
        else:
            injections['mass_2_source'] = m2

        if chi_eff or chi_p:
            injections['chi_eff'] = eval_chi_eff(q, a1, a2, c1, c2)
            if chi_p:
                assert chi_eff
                injections['chi_p'] = eval_chi_p(
                    q, a1, a2, np.arccos(c1), np.arccos(c2),
                )
                prior_iso_eff = prior_chieff_chip_isotropic(
                    injections['chi_eff'], injections['chi_p'], q,
                )
            else:
                prior_iso_eff = chi_effective_prior_from_isotropic_spins(
                    injections['chi_eff'], q,
                )
            prior_iso_spin = 1 / (2 * 2 * np.pi) ** 2
            prior = prior * prior_iso_eff / prior_iso_spin
        else:
            injections['a_1'] = a1
            injections['a_2'] = a2
            injections['cos_tilt_1'] = c1
            injections['cos_tilt_2'] = c2
            prior *= (2 * np.pi) ** 2 # fixed population model in phi1, phi2

        injections['weight'] = 1 / prior

        for key in sorted(set(extra_keys)):
            injections[key] = prior if key == 'prior' else d[key][found]

        injections = {key: np.array(injections[key]) for key in injections}

    return injections
