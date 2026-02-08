from glob import glob
import os

import bilby
from gwpopulation_pipe.analytic_spin_prior import \
    chi_effective_prior_from_isotropic_spins
import h5py
import numpy as np
from tqdm import tqdm
import wcosmo; wcosmo.disable_units()


def eval_chi_eff(mass_ratio, a_1, a_2, cos_tilt_1, cos_tilt_2):
    return \
        (a_1 * cos_tilt_1 + mass_ratio * a_2 * cos_tilt_2) / (1 + mass_ratio)


def standard_prior(redshift):
    return bilby.gw.prior.UniformSourceFrame(
        minimum = np.min(redshift),
        maximum = np.max(redshift),
        cosmology = 'Planck15_LAL',
        name = 'redshift',
    ).prob(redshift) * (1 + redshift) ** 2


def get_events(catalog = 'gwtc4', min_ifar = 1, min_mass = 3):
    url = (
        'https://gwosc.org/eventapi/ascii/query/show?release='
        'GWTC-1-confident,GWTC-1-marginal,'
        'GWTC-2.1-confident,GWTC-2.1-marginal,'
        'GWTC-3-confident,GWTC-3-marginal'
    )
    if catalog == 'gwtc4':
        url += ',GWTC-4.0'
    url += f'&min-mass-2-source={min_mass}&max-far={1 / min_ifar}'
    os.system(f'wget -O ./events.txt "{url}"')
    events = sorted(str(event) for event in np.loadtxt(
        './events.txt', dtype = str, skiprows = 1, usecols = 1,
    ))
    os.system('rm ./events.txt')
    return events


def get_posteriors_stacked(
    path,
    catalog = 'gwtc4',
    exclude = [],
    bbh = True,
    min_ifar = 1,
    min_mass = 3,
    mass_ratio = False,
    chi_eff = False,
    extra_keys = [],
    stack = True,
):
    catalog = catalog.lower()
    assert catalog in ('gwtc3', 'gwtc4')

    exclude += [
    #     'GW190426_152155', # FAR?
    #     'GW190531_023648', # FAR?
        'GW230518_125908', # NSBH ER
    #     'GW230630_070659', # DQ
    #     'GW231002_143916', # FAR=1
    #     'GW240422_213513', # FAR?
    ]
    if bbh:
        exclude += [
            'GW170817', # _124104', # BNS
            'GW190425', # _081805', # BNS? NSBH?
            'GW190814', # _211039', # NSBH?
            'GW190917_114630', # NSBH?
            'GW200105_162426', # NSBH FAR?
            'GW200115_042309', # NSBH
            'GW230529_181500', # NSBH
        ]
    exclude = sorted(set(exclude))

    events = get_events(catalog, min_ifar, min_mass)

    paths = [
        f'{path}/lvk-data/{gwtc}/PE'
        for gwtc in ('GWTC-1', 'GWTC-2', 'GWTC-2.1', 'GWTC-3')
    ]
    if catalog == 'gwtc4':
        paths.append(f'{path}/lvk-data/GWTC-4/PE')

    keys = ['redshift', 'mass_1_source']
    if mass_ratio:
        keys.append('mass_ratio')
    else:
        keys.append('mass_2_source')
    if chi_eff:
        keys.append('chi_eff')
    else:
        keys += ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']
    keys = sorted(set(keys + list(extra_keys)))

    posteriors = {}

    for event in tqdm(events):
        if event == 'GW190521': # event with repeated date without time code
            event = 'GW190521_030229'

        if any(event in excluded for excluded in exclude):
            print(f'excluding {event}: in exclude list')
            continue

        files = sorted(
            file for path in paths for file in glob(f'{path}/*{event}*')
        )
        if len(files) == 0:
            exclude.append(event)
            print(f'excluding {event}: no file found')
            continue
        file = files[0]

        posteriors[event] = {}

        with h5py.File(file, 'r') as f:

            if event == 'GW170817':
                analysis = 'IMRPhenomPv2NRT_highSpin_posterior'
                label = 'IMRPhenomPv2NRT_highSpin'

                cosmo = bilby.gw.cosmology.get_cosmology('Planck15_LAL')
                cosmo = wcosmo.FlatLambdaCDM(
                    H0 = cosmo.H0.value, Om0 = cosmo.Om0,
                )

                d = f[analysis]['luminosity_distance_Mpc']
                z = wcosmo.z_at_value(cosmo.luminosity_distance, d)

                m1 = f[analysis]['m1_detector_frame_Msun'] / (1 + z)
                m2 = f[analysis]['m2_detector_frame_Msun'] / (1 + z)
                q = m2 / m1

                posteriors[event][label] = dict(
                    redshift = z,
                    mass_1_source = m1,
                )

                if mass_ratio:
                    posteriors[event][label]['mass_ratio'] = q
                else:
                    posteriors[event][label]['mass_2_source'] = m2

                if chi_eff:
                    posteriors[event][label]['chi_eff'] = eval_chi_eff(
                        q,
                        f[analysis]['spin1'],
                        f[analysis]['spin2'],
                        f[analysis]['costilt1'],
                        f[analysis]['costilt2'],
                    )
                else:
                    posteriors[event][label]['a_1'] = f[analysis]['spin1']
                    posteriors[event][label]['a_2'] = f[analysis]['spin2']
                    posteriors[event][label]['cos_tilt_1'] = \
                        f[analysis]['costilt1']
                    posteriors[event][label]['cos_tilt_2'] = \
                        f[analysis]['costilt2']

            else:
                if event == 'GW190425':
                    analyses = 'C01:IMRPhenomPv2_NRTidal:HighSpin',
                elif event == 'GW230529_181500':
                    analyses = 'C00:Mixed:HighSpin',
                else:
                    if any('NRSur' in analysis for analysis in f):
                        analyses = 'NRSur',
                    else:
                        analyses = 'EOBNR', 'PhenomXPHM'

                analyses = [
                    analysis for analysis in f
                    if any(waveform in analysis for waveform in analyses)
                ]

                for analysis in analyses:
                    posteriors[event][analysis] = {
                        key: f[analysis]['posterior_samples'][key][()]
                        for key in keys
                    }

        norm = 0

        for analysis in posteriors[event]:
            prior = standard_prior(posteriors[event][analysis]['redshift'])
            if mass_ratio:
                prior *= posteriors[event][analysis]['mass_1_source']
            if chi_eff:
                if mass_ratio:
                    q = posteriors[event][analysis]['mass_ratio']
                else:
                    q = (
                        posteriors[event][analysis]['mass_2_source'] /
                        posteriors[event][analysis]['mass_1_source']
                    )
                prior *= chi_effective_prior_from_isotropic_spins(
                    posteriors[event][analysis]['chi_eff'], q,
                )

            posteriors[event][analysis]['weight'] = 1 / prior

            posteriors[event][analysis]['weight'] /= prior.size
            norm += posteriors[event][analysis]['weight'].sum()

        for analysis in posteriors[event]:
            posteriors[event][analysis]['weight'] /= norm

    events = list(posteriors)

    if not stack:
        return posteriors, events, sorted(set(exclude))

    posteriors_stacked = {key: [] for key in keys + ['weight', 'total']}
    for event in posteriors:
        total = 0
        for analysis in posteriors[event]:
            total += posteriors[event][analysis]['weight'].size
            for key in posteriors[event][analysis]:
                posteriors_stacked[key] = np.concatenate([
                    posteriors_stacked[key], posteriors[event][analysis][key],
                ])
        posteriors_stacked['total'].append(total)
    posteriors_stacked['total'] = np.array(posteriors_stacked['total'])

    return posteriors_stacked, events, sorted(set(exclude))


def get_posteriors(
    path,
    catalog = 'gwtc4',
    exclude = [],
    bbh = True,
    min_ifar = 1,
    min_mass = 3,
    mass_ratio = False,
    chi_eff = False,
    extra_keys = [],
    downsample = True,
):
    catalog = catalog.lower()
    assert catalog in ('gwtc3', 'gwtc4')

    exclude += [
    #     'GW190426_152155', # FAR?
    #     'GW190531_023648', # FAR?
        'GW230518_125908', # NSBH ER
    #     'GW230630_070659', # DQ
    #     'GW231002_143916', # FAR=1
    #     'GW240422_213513', # FAR?
    ]
    if bbh:
        exclude += [
            'GW170817', # _124104', # BNS
            'GW190425', # _081805', # BNS? NSBH?
            'GW190814', # _211039', # NSBH?
            'GW190917_114630', # NSBH?
            'GW200105_162426', # NSBH FAR?
            'GW200115_042309', # NSBH
            'GW230529_181500', # NSBH
        ]
    exclude = sorted(set(exclude))

    events = get_events(catalog, min_ifar, min_mass)

    paths = [
        f'{path}/lvk-data/{gwtc}/PE'
        for gwtc in ('GWTC-1', 'GWTC-2', 'GWTC-2.1', 'GWTC-3')
    ]
    if catalog == 'gwtc4':
        paths.append(f'{path}/lvk-data/GWTC-4/PE')

    keys = ['redshift', 'mass_1_source']
    if mass_ratio:
        keys.append('mass_ratio')
    else:
        keys.append('mass_2_source')
    if chi_eff:
        keys.append('chi_eff')
    else:
        keys += ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']
    keys = sorted(set(keys + list(extra_keys)))

    posteriors = {}

    for event in tqdm(events):
        if event == 'GW190521': # event with repeated date without time code
            event = 'GW190521_030229'

        if any(event in excluded for excluded in exclude):
            print(f'excluding {event}: in exclude list')
            continue

        files = sorted(
            file for path in paths for file in glob(f'{path}/*{event}*')
        )
        if len(files) == 0:
            exclude.append(event)
            print(f'excluding {event}: no file found')
            continue
        file = files[0]

        with h5py.File(file, 'r') as f:

            if event == 'GW170817':
                analysis = 'IMRPhenomPv2NRT_highSpin_posterior'

                cosmo = bilby.gw.cosmology.get_cosmology('Planck15_LAL')
                cosmo = wcosmo.FlatLambdaCDM(
                    H0 = cosmo.H0.value, Om0 = cosmo.Om0,
                )

                d = f[analysis]['luminosity_distance_Mpc']
                z = wcosmo.z_at_value(cosmo.luminosity_distance, d)

                m1 = f[analysis]['m1_detector_frame_Msun'] / (1 + z)
                m2 = f[analysis]['m2_detector_frame_Msun'] / (1 + z)
                q = m2 / m1

                posteriors[event] = dict(
                    luminosity_distance = d,
                    redshift = z,
                    mass_1_source = m1,
                )

                if mass_ratio:
                    posteriors[event]['mass_ratio'] = q
                else:
                    posteriors[event]['mass_2_source'] = m2

                if chi_eff:
                    posteriors[event]['chi_eff'] = eval_chi_eff(
                        q,
                        f[analysis]['spin1'],
                        f[analysis]['spin2'],
                        f[analysis]['costilt1'],
                        f[analysis]['costilt2'],
                    )
                else:
                    posteriors[event]['a_1'] = f[analysis]['spin1']
                    posteriors[event]['a_2'] = f[analysis]['spin2']
                    posteriors[event]['cos_tilt_1'] = f[analysis]['costilt1']
                    posteriors[event]['cos_tilt_2'] = f[analysis]['costilt2']

            else:
                if event == 'GW190425':
                    analysis = 'C01:IMRPhenomPv2_NRTidal:HighSpin'
                elif event == 'GW230529_181500':
                    analysis = 'C00:Mixed:HighSpin'
                else:
                    analysis = None
                    for key in 'C00:NRSur7dq4', 'C00:Mixed', 'C01:Mixed':
                        if key in f:
                            analysis = key
                            break
                    assert analysis is not None

                posteriors[event] = {
                    key: f[analysis]['posterior_samples'][key][()]
                    for key in keys
                }

        posteriors[event]['file'] = file
        posteriors[event]['analysis'] = analysis

    events = list(posteriors)

    if downsample is not False:
        if downsample is True:
            max_samples = np.inf
            for event in posteriors:
                max_samples = min(
                    max_samples, posteriors[event]['redshift'].size,
                )
        else:
            assert type(downsample) is int
            max_samples = int(downsample)

        for event in posteriors:
            idxs = np.random.choice(
                posteriors[event]['redshift'].size,
                min(posteriors[event]['redshift'].size, max_samples),
                replace = False,
            )
            for key in set(posteriors[event]) - {'file', 'analysis'}:
                posteriors[event][key] = posteriors[event][key][idxs]

    for event in tqdm(posteriors):
        if 'GW170817' in event:
            prior = (
                posteriors[event].pop('luminosity_distance') ** 2
                * cosmo.dDLdz(posteriors[event]['redshift'])
                * (1 + posteriors[event]['redshift']) ** 2
            )
        else:
            prior = standard_prior(posteriors[event]['redshift'])
        if mass_ratio:
            prior *= posteriors[event]['mass_1_source']
        if chi_eff:
            if mass_ratio:
                q = posteriors[event]['mass_ratio']
            else:
                q = (
                    posteriors[event]['mass_2_source'] /
                    posteriors[event]['mass_1_source']
                )
            prior *= chi_effective_prior_from_isotropic_spins(
                posteriors[event]['chi_eff'], q,
            )
        posteriors[event]['weight'] = 1 / prior

    if downsample is not False:
        for event in posteriors:
            posteriors[event]['total'] = len(posteriors[event]['weight'])
            if posteriors[event]['total'] < max_samples:
                for key in keys:
                    posteriors[event][key] = np.concatenate((
                        posteriors[event][key],
                        np.ones(max_samples - posteriors[event]['total']),
                    ))
                posteriors[event]['weight'] = np.concatenate((
                    posteriors[event]['weight'],
                    np.zeros(max_samples - posteriors[event]['total']),
                ))

        posteriors = {
            k: np.array([posteriors[event][k] for event in posteriors])
            for k in keys + ['weight', 'total']
        }

    return posteriors, events, sorted(set(exclude))


def get_injections(
    path,
    catalog = 'gwtc4',
    min_ifar = 1,
    min_snr = 10,
    mass_ratio = False,
    chi_eff = False,
    extra_keys = [],
):
    if catalog == 'gwtc3':
        file = (
            f'{path}/lvk-data/GWTC-4/VT/'
            'mixture-semi_o1_o2-real_o3-cartesian_spins_20250503134659UTC.hdf'
        )
    elif catalog == 'gwtc4':
        file = (
            f'{path}/lvk-data/GWTC-4/VT/'
            'mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf'
        )

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
    
        if chi_eff:
            injections['chi_eff'] = eval_chi_eff(q, a1, a2, c1, c2)
            prior_iso_spin = 1 / (2 * 2 * np.pi) ** 2
            prior_iso_eff = chi_effective_prior_from_isotropic_spins(
                injections['chi_eff'], q,
            )
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
