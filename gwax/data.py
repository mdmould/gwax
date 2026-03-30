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

from gwax.cosmology import source_to_detector


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
    if not er and not bbh:
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
    # elif catalog == 'GWTC-1': # this option should never end up selected
    #     return ['IMRPhenomPv2_posterior', 'SEOBNRv3_posterior']
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

def get_event(path, event, keys):
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
    catalog, file = get_event_catalog_and_file(path, event)
    data = dict(catalog = catalog, file = file)
    with h5py.File(file) as f:
        analyses = waveform_priority(event, catalog, f)
        for analysis in analyses:
            if catalog == 'GWTC-1':
                samples = f[analysis]
                data[analysis] = {key: samples[swap[key]] for key in keys}
            else:
                samples = f[analysis]['posterior_samples']
                data[analysis] = {key: samples[key] for key in keys}
    return data

def downsample_posterior(posterior, total):
    analyses = sorted(set(posterior) - {'catalog', 'file'})
    keys = list(posterior[analyses[0]].keys())
    new_posterior = {key: [] for key in keys}
    num_samples = int(np.ceil(total / len(analyses)))
    for analysis in analyses:
        total_analysis = posterior[analysis][keys[0]].size
        idx = np.random.choice(total_analysis, num_samples, replace = False)
        for key in keys:
            new_posterior[key] = np.concatenate(
                [new_posterior[key], posterior[analysis][key][idx]],
            )
    for key in keys:
        new_posterior[key] = new_posterior[key][:total]
    return new_posterior


def prior_euclidean(luminosity_distance):
    return luminosity_distance ** 2

def prior_comoving_source(luminosity_distance, H0, Om0):
    redshift = wcosmo.z_at_value(
        wcosmo.luminosity_distance, luminosity_distance, H0 = H0, Om0 = Om0,
    )
    dd_dz = wcosmo.dDLdz(redshift, H0, Om0)
    dv_dz = wcosmo.differential_comoving_volume(redshift, H0, Om0)
    return dv_dz / (1 + redshift) / dd_dz

def compute_initial_prior(data, catalog):
    H0 = 67.9
    Om0 = 0.3065
    if catalog == 'GWTC-4':
        return prior_comoving_source(data['luminosity_distance'], H0, Om0)
    else:
        return prior_euclidean(data['luminosity_distance'])


def eval_chi_eff(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    a_1z = a_1 * cos_tilt_1
    a_2z = a_2 * cos_tilt_2
    return (a_1z + mass_ratio * a_2z) / (1 + mass_ratio)

def eval_chi_p(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    a_1p = a_1 * np.sin(np.arccos(cos_tilt_1))
    a_2p = a_2 * np.sin(np.arccos(cos_tilt_2))
    coeff = mass_ratio * (4 * mass_ratio + 3) / (4 + 3 * mass_ratio)
    return np.maximum(a_1p, a_2p * coeff)

def convert_effective_spin_posterior(data, chi_eff, chi_p):
    if chi_eff or chi_p:
        mass_ratio = data['mass_2'] / data['mass_1']
        a_1 = data.pop('a_1')
        a_2 = data.pop('a_2')
        cos_tilt_1 = data.pop('cos_tilt_1')
        cos_tilt_2 = data.pop('cos_tilt_2')
        data['chi_eff'] = eval_chi_eff(
            mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2,
        )
        if chi_p:
            assert chi_eff
            data['chi_p'] = eval_chi_p(
                mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2,
            )
            data['weight'] /= prior_chieff_chip_isotropic(
                data['chi_eff'], data['chi_p'], mass_ratio,
            )
        else:
            data['weight'] /= chi_effective_prior_from_isotropic_spins(
                data['chi_eff'], mass_ratio,
            )
    return data


def get_posteriors(
    path,
    catalog = 'GWTC-4',
    min_ifar = 1,
    bbh = True,
    er = False,
    exclude = [],
    chi_eff = False,
    chi_p = False,
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
        try:
            posteriors[event] = get_event(path, event, keys)
        except:
            exclude.append(event)
            print(f'Could not get sample for {event}, excluding...')

    events = list(posteriors)
    exclude = sorted(set(exclude))

    if downsample:
        total = np.inf
        for event in posteriors:
            analyses = set(posteriors[event]) - {'catalog', 'file'}
            min_samples = np.inf
            for analysis in analyses:
                num_samples = posteriors[event][analysis]['a_1'].size
                min_samples = min(min_samples, num_samples)
            total = min(total, min_samples * len(analyses))

        new_posteriors = {key: [] for key in keys}
        for event in posteriors:
            new_posterior = downsample_posterior(posteriors[event], total)
            for key in keys:
                new_posteriors[key].append(new_posterior[key])
        new_posteriors['total'] = np.array([total] * len(posteriors))
        new_posteriors = {
            key: np.array(new_posteriors[key]) for key in new_posteriors
        }

        priors = []
        for event, luminosity_distance in zip(
            events, new_posteriors['luminosity_distance'],
        ):
            prior = compute_initial_prior(
                dict(luminosity_distance = luminosity_distance),
                posteriors[event]['catalog'],
            )
            priors.append(prior)
        new_posteriors['weight'] = 1 / np.array(priors)

        new_posteriors = convert_effective_spin_posterior(
            new_posteriors, chi_eff, chi_p,
        )

        new_posteriors['weight'] /= np.sum(
            new_posteriors['weight'], axis = 1, keepdims = True,
        )

        return new_posteriors, events, exclude

    for event in posteriors:
        analyses = set(posteriors[event]) - {'catalog', 'file'}
        for analysis in analyses:
            prior = compute_initial_prior(
                posteriors[event][analysis], posteriors[event]['catalog'],
            )
            posteriors[event][analysis]['weight'] = 1 / prior / prior.size
            posteriors[event][analysis] = convert_effective_spin_posterior(
                posteriors[event][analysis], chi_eff, chi_p,
            )

    if not stack:
        return posteriors, events, exclude

    keys = ['luminosity_distance', 'mass_1', 'mass_2']
    if chi_eff:
        keys.append('chi_eff')
        if chi_p:
            keys.append('chi_p')
    else:
        keys += ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']
    new_posteriors = {key: [] for key in keys + ['weight', 'total']}
    for event in posteriors:
        analyses = sorted(set(posteriors[event]) - {'catalog', 'file'})
        for analysis in analyses:
            for key in keys:
                new_posteriors[key] = np.concatenate([
                    new_posteriors[key], posteriors[event][analysis][key],
                ])
        weight = np.concatenate([
            posteriors[event][analysis]['weight'] for analysis in analyses
        ])
        weight /= weight.sum()
        new_posteriors['weight'] = np.concatenate(
            [new_posteriors['weight'], weight],
        )
        new_posteriors['total'].append(weight.size)
    new_posteriors['total'] = np.array(new_posteriors['total'])

    return new_posteriors, events, exclude


def get_injections(
    path,
    catalog = 'GWTC-4',
    min_ifar = 1,
    min_snr = 10,
    chi_eff = False,
    chi_p = False,
):
    file = f'{path}/lvk-data/GWTC-4/VT/mixture-semi_o1_o2-real_o3'
    if catalog == 'GWTC-3':
        file += '-cartesian_spins_20250503134659UTC.hdf'
    elif catalog == 'GWTC-4':
        file += '_o4a-cartesian_spins_20250503134659UTC.hdf'
    print(file)
    return _get_injections(
        path + '/' + file, catalog, min_ifar, min_snr, chi_eff, chi_p,
    )

def _get_injections(
    file,
    min_ifar = 1,
    min_snr = 10,
    chi_eff = False,
    chi_p = False,
):
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

        m1z, m2z, dl, jac = source_to_detector(m1, m2, z, 67.9, 0.3065)
        injections['luminosity_distance'] = dl
        injections['mass_1'] = m1z
        injections['mass_2'] = m2z
        prior *= jac

        if chi_eff or chi_p:
            injections['chi_eff'] = eval_chi_eff(q, a1, a2, c1, c2)
            if chi_p:
                assert chi_eff
                injections['chi_p'] = eval_chi_p(q, a1, a2, c1, c2)
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

    injections = {key: np.array(injections[key]) for key in injections}

    return injections
