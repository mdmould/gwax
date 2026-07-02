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
import h5ify


def get_events_list(catalog, min_ifar = 1, bbh = True, er = False, path = None):
    catalogs = (
        'GWTC-1',
        # 'GWTC-2',
        'GWTC-2.1',
        'GWTC-3',
        'GWTC-4',
        'GWTC-4.1',
        'GWTC-5',
    )
    assert catalog in catalogs

    cache_file = f'{path}/events'
    cache_file += f'-{catalog}'
    cache_file += f'-ifar{min_ifar}'
    if bbh: cache_file += '-bbh'
    if er: cache_file += '-er'
    cache_file += '.txt'

    if path is not None and os.path.exists(cache_file):
        print('loading cache:', cache_file)
        events = np.loadtxt(cache_file, dtype = str)
        return sorted(map(str, events))

    url = 'https://gwosc.org/eventapi/ascii/query/show?release='
    url += ','.join(
        [
            'GWTC-1-confident,GWTC-1-marginal',
            # 'GWTC-2',
            'GWTC-2.1-confident,GWTC-2.1-marginal',
            'GWTC-3-confident,GWTC-3-marginal',
            'GWTC-4.0',
            'GWTC-4.1',
            'GWTC-5.0',
        ][:catalogs.index(catalog) + 1]
    )
    url += f'&min-mass-2-source={3 if bbh else 0}&max-far={1 / min_ifar}'

    os.system(f'wget -O {cache_file} "{url}"')
    events = np.loadtxt(cache_file, dtype = str, skiprows = 1, usecols = 1)
    events = sorted(map(str, np.unique(events)))

    if not er and not bbh and 'GW230518_125908' in events:
        events.remove('GW230518_125908')

    if path is not None:
        np.savetxt(cache_file, events, fmt = '%s')
    else:
        os.system(f'rm {cache_file}')

    return events

def get_event_file(path, catalog, event):
    if catalog == 'GWTC-2':
        files = glob(f'{path}/lvk-data/{catalog}/PE/{event}.h5')
    elif catalog in ['GWTC-2.1', 'GWTC-3']:
        files = glob(f'{path}/lvk-data/{catalog}/PE/*{event}*_nocosmo.h5')
    else: # GWTC-1, GWTC-4, GWTC-4.1, GWTC-5
        files = glob(f'{path}/lvk-data/{catalog}/PE/*{event}*.hdf5')
    if event == 'GW190521':
        files = [file for file in files if 'GW190521_074359' not in file]
    assert len(files) == 1
    return files[0]

def get_event_catalog_and_file(path, event):
    catalogs = (
        'GWTC-1',
        'GWTC-2',
        'GWTC-2.1',
        'GWTC-3',
        'GWTC-4',
        'GWTC-4.1',
        'GWTC-5',
    )
    files = {}
    for catalog in catalogs:
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
    elif 'GWTC-4' in files and 'GWTC-4.1' in files:
        catalog = 'GWTC-4.1'
    else:
        assert len(files) == 1
        catalog = list(files)[0]
    return catalog, files[catalog]

def default_waveform_priority(event, catalog, analyses):
    event_specific = dict(
        GW170817 = ['IMRPhenomPv2NRT_highSpin_posterior'],
        GW190425 = ['C01:IMRPhenomPv2_NRTidal:HighSpin'],
        GW191219_163120 = ['C01:IMRPhenomXPHM:HighSpin', 'C01:SEOBNRv4PHM'],
        GW200115_042309 = ['C01:IMRPhenomXPHM:HighSpin', 'C01:SEOBNRv4PHM'],
        GW230518_125908 = ['C00:IMRPhenomXPHM-SpinTaylor'],
        GW230529_181500 = [
            'C00:IMRPhenomXPHM:HighSpin', 'C00:SEOBNRv5PHM:HighSpin',
        ],
        GW240925_005809 = ['C01:IMRPhenomXPHM-SpinTaylor', 'C01:SEOBNRv5PHM'],
    )
    if event in event_specific:
        return event_specific[event]
    elif catalog == 'GWTC-1': # this option should never end up selected
        return ['IMRPhenomPv2_posterior', 'SEOBNRv3_posterior']
    elif catalog == 'GWTC-2': # only selected when NRSur is available
        return ['C01:NRSur7dq4']
    elif catalog == 'GWTC-2.1':
        if 'C01:SEOBNRv4PHM' in analyses:
            return ['C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']
        else:
            return ['C01:IMRPhenomXPHM']
    elif catalog == 'GWTC-3':
        return ['C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']
    elif catalog in ('GWTC-4', 'GWTC-4.1', 'GWTC-5'):
        if 'C00:NRSur7dq4' in analyses:
            return ['C00:NRSur7dq4']
        else:
            return ['C00:IMRPhenomXPHM-SpinTaylor', 'C00:SEOBNRv5PHM']

def get_event(path, event, keys, waveform_priority = default_waveform_priority):
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


def distance_prior_euclidean(luminosity_distance):
    return luminosity_distance ** 2

def distance_prior_comoving(luminosity_distance, H0, Om0):
    redshift = wcosmo.z_at_value(
        wcosmo.luminosity_distance, luminosity_distance, H0 = H0, Om0 = Om0,
    )
    dd_dz = wcosmo.dDLdz(redshift, H0, Om0)
    dv_dz = wcosmo.differential_comoving_volume(redshift, H0, Om0)
    return dv_dz / (1 + redshift) / dd_dz

def compute_initial_prior(data, catalog):
    H0 = 67.9
    Om0 = 0.3065
    if catalog in ('GWTC-1', 'GWTC-2', 'GWTC-2.1', 'GWTC-3'):
        return distance_prior_euclidean(data['luminosity_distance'])
    return distance_prior_comoving(data['luminosity_distance'], H0, Om0)


def eval_chi_eff(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    a_1z = a_1 * cos_tilt_1
    a_2z = a_2 * cos_tilt_2
    return (a_1z + mass_ratio * a_2z) / (1 + mass_ratio)

def eval_chi_p(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    a_1p = a_1 * np.sin(np.arccos(cos_tilt_1))
    a_2p = a_2 * np.sin(np.arccos(cos_tilt_2))
    coeff = mass_ratio * (4 * mass_ratio + 3) / (4 + 3 * mass_ratio)
    return np.maximum(a_1p, a_2p * coeff)

def convert_effective_spin(data, chi_eff, chi_p, pop_spins = True):
    if chi_eff or chi_p:
        mass_ratio = data['mass_2'] / data['mass_1']
        a_1 = data['a_1']
        a_2 = data['a_2']
        cos_tilt_1 = data['cos_tilt_1']
        cos_tilt_2 = data['cos_tilt_2']

        data['chi_eff'] = eval_chi_eff(
            mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2,
        )

        if chi_p:
            assert chi_eff
            data['chi_p'] = eval_chi_p(
                mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2,
            )
            prior_iso_eff = prior_chieff_chip_isotropic(
                data['chi_eff'], data['chi_p'], mass_ratio,
            )

        else:
            prior_iso_eff = chi_effective_prior_from_isotropic_spins(
                data['chi_eff'], mass_ratio,
            )

        prior_iso_spin = 1 / (2 * 2 * np.pi) ** 2
        data['weight'] *= prior_iso_spin
        data['weight'] /= prior_iso_eff

        if pop_spins:
            for key in 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2':
                data.pop(key)

    else:
        p_pop_phi_1_phi_2 = 1 / (2 * np.pi) ** 2 # fixed population model in phi1, phi2
        data['weight'] *= p_pop_phi_1_phi_2

    return data


def get_posteriors(
    path,
    catalog,
    min_ifar = 1,
    bbh = True,
    er = False,
    exclude = [],
    waveform_priority = default_waveform_priority,
    chi_eff = False,
    chi_p = False,
    pop_spins = True,
    downsample = False,
    stack = False,
    cache = True,
):
    cache_file = f'{path}/lvk-data/cache/posteriors'
    if type(catalog) is str: cache_file += f'-{catalog}'
    cache_file += f'-ifar{min_ifar}'
    if bbh: cache_file += '-bbh'
    if er: cache_file += '-er'
    if len(exclude) > 0:
        cache_file += '-exclude'
        for event in exclude:
            cache_file += f'-{event}'
    if chi_eff: cache_file += '-chi_eff'
    if chi_p: cache_file += '-chi_p'
    if downsample: cache_file += '-downsample'
    if stack: cache_file += '-stack'
    cache_file += '.h5'

    if cache and os.path.exists(cache_file):
        print('loading cache:', cache_file)
        return h5ify.load(cache_file)

    if type(catalog) is str:
        events = get_events_list(
            catalog, min_ifar, bbh, er, path = f'{path}/lvk-data/cache',
        )
    else:
        events = catalog

    exclude = sorted(set(exclude))

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
            posteriors[event] = get_event(
                path, event, keys, waveform_priority = waveform_priority,
            )
        except:
            exclude.append(event)
            print(f'Could not get samples for {event}, excluding...')

    events = sorted(posteriors)
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

        new_posteriors = convert_effective_spin(
            new_posteriors, chi_eff, chi_p, pop_spins,
        )

        new_posteriors['weight'] /= np.sum(
            new_posteriors['weight'], axis = 1, keepdims = True,
        )

        # return new_posteriors, events, exclude

    else:

        for event in posteriors:
            analyses = set(posteriors[event]) - {'catalog', 'file'}
            for analysis in analyses:
                prior = compute_initial_prior(
                    posteriors[event][analysis], posteriors[event]['catalog'],
                )
                posteriors[event][analysis]['weight'] = 1 / prior / prior.size
                posteriors[event][analysis] = convert_effective_spin(
                    posteriors[event][analysis], chi_eff, chi_p, pop_spins,
                )

        new_posteriors = posteriors

        if stack:

            keys = ['luminosity_distance', 'mass_1', 'mass_2']
            if chi_eff:
                if not pop_spins:
                    keys += ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']
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
                            new_posteriors[key],
                            posteriors[event][analysis][key],
                        ])

                weight = np.concatenate([
                    posteriors[event][analysis]['weight']
                    for analysis in analyses
                ])
                weight /= weight.sum()
                new_posteriors['weight'] = np.concatenate(
                    [new_posteriors['weight'], weight],
                )
                new_posteriors['total'].append(weight.size)

            new_posteriors['total'] = np.array(new_posteriors['total'])

    new_posteriors['events'] = events
    new_posteriors['excluded'] = exclude

    if cache:
        print('saving cache:', cache_file)
        h5ify.save(cache_file, new_posteriors)

    return new_posteriors


def get_injections(
    path,
    catalog,
    min_ifar = 1,
    min_snr = 10,
    chi_eff = False,
    chi_p = False,
    pop_spins = True,
    cache = True,
):
    cache_file = f'{path}/lvk-data/cache/injections'
    cache_file += f'-{catalog}'
    cache_file += f'-ifar{min_ifar}'
    cache_file += f'-snr{min_snr}'
    if chi_eff: cache_file += '-chi_eff'
    if chi_p: cache_file += '-chi_p'
    cache_file += '.h5'

    if cache and os.path.exists(cache_file):
        print('loading cache:', cache_file)
        return h5ify.load(cache_file)

    if catalog == 'GWTC-3':
        file = 'GWTC-4/VT/mixture-semi_o1_o2-real_o3-cartesian_spins_20250503134659UTC.hdf'
    elif catalog == 'GWTC-4':
        file = 'GWTC-4/VT/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf'
    elif catalog == 'GWTC-5':
        file = 'GWTC-5/VT/mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_20260410130052UTC-clipped.hdf'
    else:
        print(catalog, 'not available')

    file = f'{path}/lvk-data/{file}'
    print(file)

    injections = _get_injections(
        file, min_ifar, min_snr, chi_eff, chi_p, pop_spins,
    )

    if cache:
        print('saving cache:', cache_file)
        h5ify.save(cache_file, injections)

    return injections

def _get_injections(
    file,
    min_ifar = 1,
    min_snr = 10,
    chi_eff = False,
    chi_p = False,
    pop_spins = True,
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

    # q = m2 / m1
    # a1 = (s1x ** 2 + s1y ** 2 + s1z ** 2) ** 0.5
    # a2 = (s2x ** 2 + s2y ** 2 + s2z ** 2) ** 0.5
    # c1 = s1z / a1
    # c2 = s2z / a2
    # prior *= a1 ** 2 * a2 ** 2 # (x, y, z) -> (a, cos(theta), phi)

    (
        injections['mass_1'],
        injections['mass_2'],
        injections['luminosity_distance'],
        jac,
    ) = source_to_detector(m1, m2, z, 67.9, 0.3065)
    prior *= jac

    injections['a_1'] = (s1x ** 2 + s1y ** 2 + s1z ** 2) ** 0.5
    injections['a_2'] = (s2x ** 2 + s2y ** 2 + s2z ** 2) ** 0.5
    injections['cos_tilt_1'] = s1z / injections['a_1']
    injections['cos_tilt_2'] = s2z / injections['a_2']
    prior *= injections['a_1'] ** 2 * injections['a_2'] ** 2 # (x, y, z) -> (a, cos(theta), phi)

    injections['weight'] = 1 / prior

    injections = convert_effective_spin(injections, chi_eff, chi_p, pop_spins)

    # if chi_eff or chi_p:
    #     injections['chi_eff'] = eval_chi_eff(q, a1, a2, c1, c2)
    #     if chi_p:
    #         assert chi_eff
    #         injections['chi_p'] = eval_chi_p(q, a1, a2, c1, c2)
    #         prior_iso_eff = prior_chieff_chip_isotropic(
    #             injections['chi_eff'], injections['chi_p'], q,
    #         )
    #     else:
    #         prior_iso_eff = chi_effective_prior_from_isotropic_spins(
    #             injections['chi_eff'], q,
    #         )
    #     prior_iso_spin = 1 / (2 * 2 * np.pi) ** 2
    #     prior = prior * prior_iso_eff / prior_iso_spin
    # else:
    #     injections['a_1'] = a1
    #     injections['a_2'] = a2
    #     injections['cos_tilt_1'] = c1
    #     injections['cos_tilt_2'] = c2
    #     prior *= (2 * np.pi) ** 2 # fixed population model in phi1, phi2

    # injections['weight'] = 1 / prior

    injections = {key: np.array(injections[key]) for key in injections}

    return injections
