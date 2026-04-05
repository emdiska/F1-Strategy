"""
model.py — Tyre degradation and pit loss model
Fits degradation curves per circuit per compound from stint data.
Uses multiple regression separating fuel burn from tyre degradation.
Corrects compound pace ordering using FP2 session data,
validated against actual race pace gaps.
Constrains max stint length per compound per circuit.
Applies Pirelli circuit ratings as fallback for sparse data.
Fits piecewise tyre cliff model per compound per circuit.
Scales cliff degradation by track temperature (Arrhenius).
Enforces physical bounds on fuel rate and degradation ordering.
"""

import os
import time as time_module
import numpy as np
import pandas as pd
import fastf1
import json

os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

DATA_PATH  = 'data/stints.csv'
META_PATH  = 'data/race_metadata.csv'
MODEL_PATH = 'data/model.json'
TEMPS_PATH = 'data/track_temps.json'

COMPOUNDS       = ['SOFT', 'MEDIUM', 'HARD']
MIN_STINT_LAPS  = 6
MIN_STINTS      = 3
MAX_TYRE_AGE    = 55
MIN_DEG_RATE    = 0.005
MAX_DEG_RATE    = 1.5
DRY_COMPOUNDS   = {'SOFT', 'MEDIUM', 'HARD'}

FALLBACK_DEG = {
    'SOFT':   0.080,
    'MEDIUM': 0.045,
    'HARD':   0.020,
}

# Fuel rate physical bounds (s/lap per race lap)
# Based on 1.6 kg/lap burn rate × 0.028-0.040 s/kg
# Allow wider range for circuit variation
FUEL_RATE_MIN = -0.150  # most sensitive circuit
FUEL_RATE_MAX = -0.010  # least sensitive circuit
FUEL_RATE_FALLBACK = -0.030  # RAE 2010 reference value

PIRELLI_RATINGS = {
    'Bahrain Grand Prix':         {'tyre_stress': 4, 'lateral': 3, 'abrasion': 4, 'traction': 4, 'braking': 3, 'track_evolution': 3, 'downforce': 3},
    'Saudi Arabian Grand Prix':   {'tyre_stress': 3, 'lateral': 3, 'abrasion': 2, 'traction': 3, 'braking': 4, 'track_evolution': 4, 'downforce': 3},
    'Australian Grand Prix':      {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 4, 'downforce': 3},
    'Japanese Grand Prix':        {'tyre_stress': 5, 'lateral': 5, 'abrasion': 4, 'traction': 3, 'braking': 3, 'track_evolution': 2, 'downforce': 3},
    'Chinese Grand Prix':         {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 4, 'downforce': 3},
    'Miami Grand Prix':           {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 4, 'downforce': 3},
    'Emilia Romagna Grand Prix':  {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 3},
    'Monaco Grand Prix':          {'tyre_stress': 2, 'lateral': 2, 'abrasion': 1, 'traction': 4, 'braking': 4, 'track_evolution': 5, 'downforce': 5},
    'Canadian Grand Prix':        {'tyre_stress': 3, 'lateral': 2, 'abrasion': 2, 'traction': 3, 'braking': 5, 'track_evolution': 4, 'downforce': 2},
    'Spanish Grand Prix':         {'tyre_stress': 4, 'lateral': 4, 'abrasion': 3, 'traction': 4, 'braking': 3, 'track_evolution': 2, 'downforce': 4},
    'Austrian Grand Prix':        {'tyre_stress': 4, 'lateral': 4, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 4},
    'Styrian Grand Prix':         {'tyre_stress': 4, 'lateral': 4, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 4},
    'British Grand Prix':         {'tyre_stress': 5, 'lateral': 5, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 4},
    'Hungarian Grand Prix':       {'tyre_stress': 3, 'lateral': 4, 'abrasion': 2, 'traction': 4, 'braking': 3, 'track_evolution': 4, 'downforce': 5},
    'Belgian Grand Prix':         {'tyre_stress': 4, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 3},
    'Dutch Grand Prix':           {'tyre_stress': 5, 'lateral': 5, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 4, 'downforce': 5},
    'Italian Grand Prix':         {'tyre_stress': 3, 'lateral': 2, 'abrasion': 3, 'traction': 2, 'braking': 5, 'track_evolution': 3, 'downforce': 1},
    'Azerbaijan Grand Prix':      {'tyre_stress': 2, 'lateral': 2, 'abrasion': 2, 'traction': 3, 'braking': 5, 'track_evolution': 5, 'downforce': 2},
    'Singapore Grand Prix':       {'tyre_stress': 3, 'lateral': 3, 'abrasion': 2, 'traction': 4, 'braking': 4, 'track_evolution': 5, 'downforce': 5},
    'United States Grand Prix':   {'tyre_stress': 4, 'lateral': 4, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 4},
    'Mexico City Grand Prix':     {'tyre_stress': 3, 'lateral': 3, 'abrasion': 2, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 5},
    'Brazilian Grand Prix':       {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 3},
    'São Paulo Grand Prix':       {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 3},
    'Las Vegas Grand Prix':       {'tyre_stress': 2, 'lateral': 2, 'abrasion': 2, 'traction': 3, 'braking': 4, 'track_evolution': 5, 'downforce': 2},
    'Qatar Grand Prix':           {'tyre_stress': 5, 'lateral': 5, 'abrasion': 4, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 4},
    'Abu Dhabi Grand Prix':       {'tyre_stress': 3, 'lateral': 3, 'abrasion': 2, 'traction': 3, 'braking': 3, 'track_evolution': 4, 'downforce': 3},
    'French Grand Prix':          {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 3},
    'Portuguese Grand Prix':      {'tyre_stress': 3, 'lateral': 3, 'abrasion': 2, 'traction': 3, 'braking': 3, 'track_evolution': 4, 'downforce': 3},
    'Russian Grand Prix':         {'tyre_stress': 2, 'lateral': 2, 'abrasion': 2, 'traction': 3, 'braking': 3, 'track_evolution': 5, 'downforce': 3},
    'Turkish Grand Prix':         {'tyre_stress': 4, 'lateral': 4, 'abrasion': 4, 'traction': 3, 'braking': 3, 'track_evolution': 2, 'downforce': 4},
    'Tuscan Grand Prix':          {'tyre_stress': 4, 'lateral': 4, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 3},
    'Eifel Grand Prix':           {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 3},
    'Sakhir Grand Prix':          {'tyre_stress': 4, 'lateral': 3, 'abrasion': 4, 'traction': 4, 'braking': 3, 'track_evolution': 3, 'downforce': 2},
    '70th Anniversary Grand Prix':{'tyre_stress': 5, 'lateral': 5, 'abrasion': 3, 'traction': 3, 'braking': 3, 'track_evolution': 3, 'downforce': 4},
    'German Grand Prix':          {'tyre_stress': 3, 'lateral': 3, 'abrasion': 3, 'traction': 3, 'braking': 4, 'track_evolution': 3, 'downforce': 3},
}

SEASONS = list(range(2020, 2026))


def fit_degradation_curves(df: pd.DataFrame) -> dict:
    """
    Fit degradation using multiple regression with quadratic
    tyre age term:
        lap_time = base + fuel_rate * race_lap
                        + deg_rate   * stint_lap
                        + deg_rate_2 * stint_lap²

    Enforces physical bounds on fuel_rate.
    Enforces degradation ordering: SOFT >= MEDIUM >= HARD.
    """
    df_dry = df[~df['wet_race']].copy()
    df_dry = df_dry[df_dry['tyre_age'] <= MAX_TYRE_AGE]
    df_dry = df_dry[df_dry['stint_lap'] >= 1]

    circuits = sorted(df_dry['circuit'].unique())
    model    = {}

    for circuit in circuits:
        model[circuit] = {}
        df_c      = df_dry[df_dry['circuit'] == circuit]
        dry_races = int(
            df_c[['year', 'round']].drop_duplicates().shape[0])

        for compound in COMPOUNDS:
            df_cc = df_c[df_c['compound'] == compound].copy()

            if len(df_cc) < MIN_STINT_LAPS * MIN_STINTS:
                continue

            fastest = df_cc['lap_time_raw'].min()
            df_cc   = df_cc[
                df_cc['lap_time_raw'] < fastest * 1.15]

            if len(df_cc) < MIN_STINT_LAPS * MIN_STINTS:
                continue

            y         = df_cc['lap_time_raw'].values
            race_lap  = df_cc['race_lap'].values.astype(float)
            stint_lap = df_cc['stint_lap'].values.astype(float)

            X = np.column_stack([
                np.ones(len(y)),
                race_lap,
                stint_lap,
                stint_lap ** 2
            ])

            try:
                coeffs, _, _, _ = np.linalg.lstsq(
                    X, y, rcond=None)
            except Exception:
                continue

            base       = float(coeffs[0])
            fuel_rate  = float(coeffs[1])
            deg_rate   = float(coeffs[2])
            deg_rate_2 = float(coeffs[3])

            # Fuel rate physical bounds
            # 1.6 kg/lap burn × 0.028-0.040 s/kg typical
            # Allow wider range for circuit variation
            if (fuel_rate > FUEL_RATE_MAX or
                    fuel_rate < FUEL_RATE_MIN):
                fuel_rate = FUEL_RATE_FALLBACK

            if deg_rate > MAX_DEG_RATE:
                continue

            if deg_rate < MIN_DEG_RATE:
                deg_rate   = FALLBACK_DEG[compound]
                deg_rate_2 = 0.0

            if deg_rate_2 < -0.005:
                deg_rate_2 = 0.0
            deg_rate_2 = min(deg_rate_2, 0.005)

            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            base_pace = (base
                         + fuel_rate  * 1.0
                         + deg_rate   * 1.0
                         + deg_rate_2 * 1.0)

            model[circuit][compound] = {
                'base_pace':  round(float(base_pace), 4),
                'deg_rate':   round(float(deg_rate), 4),
                'deg_rate_2': round(float(deg_rate_2), 6),
                'fuel_rate':  round(float(fuel_rate), 4),
                'r_squared':  round(float(r2), 4),
                'n_points':   int(len(df_cc)),
                'dry_races':  dry_races,
            }

    # Enforce physical degradation ordering:
    # SOFT >= MEDIUM >= HARD
    # Violations indicate sparse data overfitting
    ordering_corrections = 0
    for circuit, compounds in model.items():
        soft_deg   = compounds.get(
            'SOFT',   {}).get('deg_rate', None)
        medium_deg = compounds.get(
            'MEDIUM', {}).get('deg_rate', None)
        hard_deg   = compounds.get(
            'HARD',   {}).get('deg_rate', None)

        # HARD should not exceed MEDIUM by more than 50%
        if (hard_deg and medium_deg and
                hard_deg > medium_deg * 1.5):
            compounds['HARD']['deg_rate']   = round(
                FALLBACK_DEG['HARD'], 4)
            compounds['HARD']['deg_rate_2'] = 0.0
            compounds['HARD']['ordering_corrected'] = True
            ordering_corrections += 1

        # MEDIUM should not exceed SOFT by more than 50%
        if (medium_deg and soft_deg and
                medium_deg > soft_deg * 1.5):
            compounds['MEDIUM']['deg_rate']   = round(
                FALLBACK_DEG['MEDIUM'], 4)
            compounds['MEDIUM']['deg_rate_2'] = 0.0
            compounds['MEDIUM']['ordering_corrected'] = True
            ordering_corrections += 1

    if ordering_corrections > 0:
        print(f'  Ordering corrections applied: '
              f'{ordering_corrections} compounds reset '
              f'to fallback deg rate')

    return model


def compute_max_stint_lengths(df: pd.DataFrame) -> dict:
    """
    Compute maximum observed stint length per circuit
    per compound. Adds 5% buffer, minimum 15 laps.
    """
    df_dry = df[~df['wet_race']].copy()
    result = {}

    for circuit in df_dry['circuit'].unique():
        df_c = df_dry[df_dry['circuit'] == circuit]
        result[circuit] = {}

        for compound in COMPOUNDS:
            df_cc = df_c[df_c['compound'] == compound]
            if df_cc.empty:
                continue
            max_observed = int(df_cc['stint_lap'].max())
            max_allowed  = max(15, int(max_observed * 1.05))
            result[circuit][compound] = max_allowed

    return result


def apply_pirelli_ratings_correction(
        deg_model: dict,
        max_stints: dict,
        n_points_threshold: int = 300,
        dry_races_threshold: int = 4) -> tuple:
    """
    Apply Pirelli circuit characterisation ratings as
    fallback correction for data-sparse circuits only.
    """
    DEG_LATERAL_SOFT    = 0.004
    DEG_LATERAL_MEDIUM  = 0.002
    DEG_ABRASION_ALL    = 0.003
    DEG_BRAKING_SOFT    = 0.002
    DEG_EVOLUTION_ALL   = -0.002
    STINT_STRESS_FACTOR = 0.10

    corrected_circuits = []

    for circuit, compounds in deg_model.items():
        ratings = PIRELLI_RATINGS.get(circuit)
        if not ratings:
            continue

        dry_races = max(
            (c.get('dry_races', 99)
             for c in compounds.values()),
            default=99)
        needs_correction = dry_races < dry_races_threshold

        for compound, params in compounds.items():
            n_pts = params.get('n_points', 999)
            compound_needs = (
                needs_correction or
                n_pts < n_points_threshold)

            if not compound_needs:
                continue

            r         = ratings
            delta_deg = 0.0

            lat_delta = r['lateral'] - 3
            if compound == 'SOFT':
                delta_deg += lat_delta * DEG_LATERAL_SOFT
            elif compound == 'MEDIUM':
                delta_deg += lat_delta * DEG_LATERAL_MEDIUM

            abr_delta  = r['abrasion'] - 3
            delta_deg += abr_delta * DEG_ABRASION_ALL

            brk_delta = r['braking'] - 3
            if compound == 'SOFT':
                delta_deg += brk_delta * DEG_BRAKING_SOFT

            evo_delta  = r['track_evolution'] - 3
            delta_deg += evo_delta * DEG_EVOLUTION_ALL

            original  = params['deg_rate']
            corrected = max(MIN_DEG_RATE, original + delta_deg)
            params['deg_rate']           = round(corrected, 4)
            params['pirelli_correction'] = round(delta_deg, 4)

            if circuit not in corrected_circuits:
                corrected_circuits.append(circuit)

        if circuit in max_stints:
            stress_delta = ratings['tyre_stress'] - 3
            for compound in max_stints[circuit]:
                n_pts = compounds.get(
                    compound, {}).get('n_points', 999)
                compound_needs = (
                    needs_correction or
                    n_pts < n_points_threshold)
                if not compound_needs:
                    continue
                original_max = max_stints[circuit][compound]
                factor       = 1.0 - (
                    stress_delta * STINT_STRESS_FACTOR)
                factor       = max(0.6, min(1.4, factor))
                max_stints[circuit][compound] = max(
                    10, int(original_max * factor))

    print(f'  Pirelli ratings correction applied to '
          f'{len(corrected_circuits)} circuits')

    return deg_model, max_stints


def apply_soft_cliff_prior(cliff_data: dict,
                            deg_model: dict) -> dict:
    """
    For SOFT at high-stress circuits with sparse data,
    apply physical prior: cliff at lap 8,
    phase2 = phase1 × 2.5.
    Only when n_points < 300 and
    tyre_stress >= 4 or lateral >= 4.
    """
    for circuit, compounds in cliff_data.items():
        if 'SOFT' not in compounds:
            continue

        cliff = compounds['SOFT']
        if cliff.get('has_cliff'):
            continue

        n_pts = deg_model.get(circuit, {}).get(
            'SOFT', {}).get('n_points', 999)
        if n_pts >= 300:
            continue

        ratings = PIRELLI_RATINGS.get(circuit, {})
        stress  = ratings.get('tyre_stress', 3)
        lateral = ratings.get('lateral', 3)

        if stress < 4 and lateral < 4:
            continue

        phase1 = cliff.get(
            'deg_phase1',
            deg_model[circuit]['SOFT']['deg_rate'])
        phase2 = round(phase1 * 2.5, 4)

        compounds['SOFT'] = {
            'cliff_lap':   8,
            'deg_phase1':  phase1,
            'deg_phase2':  phase2,
            'cliff_ratio': round(phase2 / phase1, 3),
            'has_cliff':   True,
            'source':      'prior (sparse data)',
        }

    return cliff_data


def fit_cliff_model(df: pd.DataFrame,
                    deg_model: dict) -> dict:
    """
    Fit two-segment piecewise linear degradation model
    per compound per circuit. Breakpoint search laps 5-25,
    minimising RSS. cliff_ratio >= 1.5 required.
    Physical plausibility caps:
    SOFT ×10, MEDIUM ×5, HARD ×3.
    """
    df_dry = df[~df['wet_race']].copy()
    df_dry = df_dry[df_dry['tyre_age'] <= MAX_TYRE_AGE]
    df_dry = df_dry[df_dry['stint_lap'] >= 1]

    cliff_data = {}

    for circuit, compounds in deg_model.items():
        cliff_data[circuit] = {}
        df_c = df_dry[df_dry['circuit'] == circuit]

        for compound in COMPOUNDS:
            if compound not in compounds:
                continue

            df_cc = df_c[df_c['compound'] == compound].copy()

            if len(df_cc) < 40:
                cliff_data[circuit][compound] = {
                    'cliff_lap':   None,
                    'deg_phase1':  compounds[compound]['deg_rate'],
                    'deg_phase2':  compounds[compound]['deg_rate'],
                    'cliff_ratio': 1.0,
                    'has_cliff':   False,
                }
                continue

            fastest = df_cc['lap_time_raw'].min()
            df_cc   = df_cc[
                df_cc['lap_time_raw'] < fastest * 1.15]

            y         = df_cc['lap_time_raw'].values
            race_lap  = df_cc['race_lap'].values.astype(float)
            stint_lap = df_cc['stint_lap'].values.astype(float)

            best_rss    = np.inf
            best_cliff  = None
            best_phase1 = compounds[compound]['deg_rate']
            best_phase2 = compounds[compound]['deg_rate']

            max_cliff = min(25, int(stint_lap.max()) - 5)
            if max_cliff < 5:
                cliff_data[circuit][compound] = {
                    'cliff_lap':   None,
                    'deg_phase1':  compounds[compound]['deg_rate'],
                    'deg_phase2':  compounds[compound]['deg_rate'],
                    'cliff_ratio': 1.0,
                    'has_cliff':   False,
                }
                continue

            for cliff in range(5, max_cliff + 1):
                n_before = np.sum(stint_lap <= cliff)
                n_after  = np.sum(stint_lap > cliff)
                if n_before < 15 or n_after < 20:
                    continue

                phase2_feat = np.maximum(0, stint_lap - cliff)
                X = np.column_stack([
                    np.ones(len(y)),
                    race_lap,
                    stint_lap,
                    phase2_feat,
                ])

                try:
                    coeffs, _, _, _ = np.linalg.lstsq(
                        X, y, rcond=None)
                except Exception:
                    continue

                phase1 = float(coeffs[2])
                delta  = float(coeffs[3])

                if phase1 < 0:
                    continue

                y_pred = X @ coeffs
                rss    = np.sum((y - y_pred) ** 2)

                if rss < best_rss:
                    best_rss    = rss
                    best_cliff  = cliff
                    best_phase1 = phase1
                    best_phase2 = phase1 + delta

            if best_cliff is None:
                cliff_data[circuit][compound] = {
                    'cliff_lap':   None,
                    'deg_phase1':  compounds[compound]['deg_rate'],
                    'deg_phase2':  compounds[compound]['deg_rate'],
                    'cliff_ratio': 1.0,
                    'has_cliff':   False,
                }
                continue

            best_phase1 = max(MIN_DEG_RATE, best_phase1)
            best_phase2 = max(best_phase1, best_phase2)

            cliff_ratio = (best_phase2 / best_phase1
                           if best_phase1 > 0 else 1.0)

            MAX_CLIFF_RATIO = {
                'SOFT':   10.0,
                'MEDIUM':  5.0,
                'HARD':    3.0,
            }
            if cliff_ratio > MAX_CLIFF_RATIO.get(compound, 5.0):
                has_cliff   = False
                cliff_ratio = 1.0
            else:
                has_cliff = cliff_ratio >= 1.5

            cliff_data[circuit][compound] = {
                'cliff_lap':   int(best_cliff) if has_cliff
                               else None,
                'deg_phase1':  round(float(best_phase1), 4),
                'deg_phase2':  round(float(best_phase2), 4),
                'cliff_ratio': round(float(cliff_ratio), 3),
                'has_cliff':   has_cliff,
            }

    cliff_data = apply_soft_cliff_prior(cliff_data, deg_model)

    n_cliffs = sum(
        1 for c in cliff_data.values()
        for v in c.values()
        if v.get('has_cliff'))
    print(f'  Cliff model fitted: {n_cliffs} compound-circuit '
          f'combinations show significant cliff (ratio >= 1.5)')

    return cliff_data


def extract_track_temperatures(
        seasons: list,
        save_path: str = TEMPS_PATH) -> dict:
    """
    Load track temperatures from cache.
    If cache missing, extract from FastF1 weather data.
    Uses race sessions, averages middle third of race.
    """
    if os.path.exists(save_path):
        print('  Loading cached track temperatures...')
        with open(save_path) as f:
            return json.load(f)

    print('  Cache not found — extracting from FastF1...')
    temps_by_circuit = {}
    session_count    = 0

    for year in seasons:
        print(f'  Processing {year}...')
        try:
            schedule = fastf1.get_event_schedule(
                year, include_testing=False)
        except Exception:
            continue

        for _, event in schedule.iterrows():
            round_num = int(event['RoundNumber'])
            if round_num == 0:
                continue

            try:
                ses = fastf1.get_session(year, round_num, 'R')
                ses.load(laps=False, telemetry=False,
                         weather=True, messages=False)
            except Exception:
                time_module.sleep(2)
                continue

            try:
                circuit = ses.event['EventName']
                weather = ses.weather_data

                if weather is None or weather.empty:
                    time_module.sleep(2)
                    continue

                n       = len(weather)
                start_i = n // 3
                end_i   = 2 * n // 3
                mid     = weather.iloc[start_i:end_i]
                if mid.empty:
                    mid = weather

                track_temp = float(mid['TrackTemp'].mean())
                air_temp   = float(mid['AirTemp'].mean())

                if circuit not in temps_by_circuit:
                    temps_by_circuit[circuit] = {
                        'track_temps': [],
                        'air_temps':   [],
                    }

                if 15.0 < track_temp < 70.0:
                    temps_by_circuit[circuit][
                        'track_temps'].append(track_temp)
                    temps_by_circuit[circuit][
                        'air_temps'].append(air_temp)

                session_count += 1

            except Exception:
                pass

            time_module.sleep(2)

    result = {}
    for circuit, data in temps_by_circuit.items():
        if data['track_temps']:
            result[circuit] = {
                'track_temp': round(
                    float(np.mean(data['track_temps'])), 1),
                'air_temp':   round(
                    float(np.mean(data['air_temps'])), 1),
                'n_races':    len(data['track_temps']),
            }

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  Extracted {session_count} races, '
          f'{len(result)} circuits')
    return result


def apply_temperature_scaling(cliff_model: dict,
                               deg_model: dict,
                               track_temps: dict) -> dict:
    """
    Scale cliff phase2 degradation rate by track temperature.

    Physical basis — Arrhenius relationship:
    Rubber compound breakdown accelerates with temperature.
    Reference temperature: 35°C.
    Above 35°C → steeper cliff.
    Below 35°C → gentler cliff.

    Scaling:
      factor = exp(k × (track_temp - T_ref) / T_ref)

    Compound sensitivity k:
      SOFT   = 0.15  (most temperature sensitive)
      MEDIUM = 0.08
      HARD   = 0.03  (thermally stable)

    Applied only to deg_phase2 (cliff region).
    Factor capped at [0.5, 2.5].
    """
    T_REF = 35.0

    COMPOUND_SENSITIVITY = {
        'SOFT':   0.15,
        'MEDIUM': 0.08,
        'HARD':   0.03,
    }

    scaled = 0

    for circuit, compounds in cliff_model.items():
        temp_data = track_temps.get(circuit)
        if not temp_data:
            continue

        track_temp = temp_data['track_temp']
        temp_delta = track_temp - T_REF

        for compound, cliff in compounds.items():
            if not cliff.get('has_cliff'):
                continue

            k      = COMPOUND_SENSITIVITY.get(compound, 0.08)
            factor = float(np.exp(k * temp_delta / T_REF))
            factor = max(0.5, min(2.5, factor))

            original_phase2 = cliff['deg_phase2']
            scaled_phase2   = round(
                float(original_phase2 * factor), 4)
            scaled_phase2   = max(
                cliff['deg_phase1'], scaled_phase2)

            cliff['deg_phase2']      = scaled_phase2
            cliff['cliff_ratio']     = round(
                scaled_phase2 / cliff['deg_phase1'], 3)
            cliff['temp_factor']     = round(float(factor), 3)
            cliff['track_temp_used'] = track_temp
            scaled += 1

    print(f'  Temperature scaling applied to '
          f'{scaled} cliff curves')

    sample_circuits = [
        'Bahrain Grand Prix',
        'Japanese Grand Prix',
        'British Grand Prix',
        'Qatar Grand Prix',
        'Monaco Grand Prix',
    ]
    print(f'  Sample temperatures:')
    for c in sample_circuits:
        t = track_temps.get(c)
        if t:
            cliff_soft = cliff_model.get(c, {}).get('SOFT', {})
            factor_str = (
                f'  SOFT factor='
                f'{cliff_soft.get("temp_factor","?")}'
                if cliff_soft.get('has_cliff') else '')
            print(f'    {c}: '
                  f'{t["track_temp"]}°C '
                  f'(n={t["n_races"]}){factor_str}')

    return cliff_model


def extract_fp_pace_gaps(seasons: list) -> dict:
    """
    Extract compound pace gaps from FP2 sessions.
    Cached to data/fp_gaps.json.
    """
    fp_gaps_path = 'data/fp_gaps.json'

    if os.path.exists(fp_gaps_path):
        print('  Loading cached FP2 gaps...')
        with open(fp_gaps_path) as f:
            return json.load(f)

    gaps_by_circuit = {}
    session_count   = 0

    for year in seasons:
        print(f'  Processing FP2 data for {year}...')
        try:
            schedule = fastf1.get_event_schedule(
                year, include_testing=False)
        except Exception as e:
            print(f'  Failed to get schedule {year}: {e}')
            continue

        for _, event in schedule.iterrows():
            round_num = int(event['RoundNumber'])
            if round_num == 0:
                continue

            try:
                ses = fastf1.get_session(year, round_num, 'FP2')
                ses.load(laps=True, telemetry=False,
                         weather=False, messages=False)
            except Exception:
                time_module.sleep(2)
                continue

            try:
                circuit  = ses.event['EventName']
                all_laps = ses.laps.copy()

                if all_laps.empty:
                    continue

                deleted_mask = (all_laps['Deleted']
                                .fillna(False)
                                .infer_objects(copy=False))

                clean = all_laps[
                    all_laps['Compound'].isin(DRY_COMPOUNDS) &
                    all_laps['PitInTime'].isna() &
                    all_laps['PitOutTime'].isna() &
                    all_laps['LapTime'].notna() &
                    deleted_mask.eq(False)
                ].copy()

                if clean.empty:
                    continue

                clean['LapTimeSec'] = (
                    clean['LapTime'].dt.total_seconds())

                fastest_overall = clean['LapTimeSec'].min()
                clean = clean[
                    clean['LapTimeSec'] < fastest_overall * 1.08]

                best = clean.groupby(
                    'Compound')['LapTimeSec'].quantile(0.10)

                if circuit not in gaps_by_circuit:
                    gaps_by_circuit[circuit] = {
                        'soft_medium': [],
                        'medium_hard': [],
                    }

                if 'SOFT' in best.index and 'MEDIUM' in best.index:
                    gap = best['MEDIUM'] - best['SOFT']
                    if 0.0 < gap < 2.0:
                        gaps_by_circuit[circuit][
                            'soft_medium'].append(float(gap))

                if 'MEDIUM' in best.index and 'HARD' in best.index:
                    gap = best['HARD'] - best['MEDIUM']
                    if 0.0 < gap < 2.0:
                        gaps_by_circuit[circuit][
                            'medium_hard'].append(float(gap))

                session_count += 1

            except Exception as e:
                print(f'    Error: {e}')

            time_module.sleep(3)

    result = {}
    for circuit, gaps in gaps_by_circuit.items():
        result[circuit] = {}
        if gaps['soft_medium']:
            result[circuit]['SOFT_vs_MEDIUM'] = round(
                float(np.mean(gaps['soft_medium'])), 3)
        if gaps['medium_hard']:
            result[circuit]['MEDIUM_vs_HARD'] = round(
                float(np.mean(gaps['medium_hard'])), 3)

    with open(fp_gaps_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  FP2 gaps extracted: '
          f'{session_count} sessions, '
          f'{len(result)} circuits')

    return result


def extract_race_pace_gaps(df: pd.DataFrame) -> dict:
    """
    Extract compound pace gaps from race stint data.
    Used to validate and correct FP2 gaps when inflated.
    """
    df_dry = df[~df['wet_race']].copy()
    result = {}

    for circuit in df_dry['circuit'].unique():
        df_c    = df_dry[df_dry['circuit'] == circuit]
        gaps_sm = []
        gaps_mh = []

        for (year, rnd), race in df_c.groupby(
                ['year', 'round']):
            best = race.groupby(
                'compound')['lap_time_fuel_adj'].quantile(0.10)

            if 'SOFT' in best.index and 'MEDIUM' in best.index:
                gap = best['MEDIUM'] - best['SOFT']
                if 0.05 < gap < 1.5:
                    gaps_sm.append(float(gap))

            if 'MEDIUM' in best.index and 'HARD' in best.index:
                gap = best['HARD'] - best['MEDIUM']
                if 0.05 < gap < 1.0:
                    gaps_mh.append(float(gap))

        if gaps_sm or gaps_mh:
            result[circuit] = {}
            if gaps_sm:
                result[circuit]['SOFT_vs_MEDIUM'] = round(
                    float(np.median(gaps_sm)), 3)
            if gaps_mh:
                result[circuit]['MEDIUM_vs_HARD'] = round(
                    float(np.median(gaps_mh)), 3)

    print(f'  Race pace gaps extracted for '
          f'{len(result)} circuits')
    return result


def apply_fp_pace_gaps(deg_model: dict,
                       fp_gaps: dict,
                       race_gaps: dict = None) -> dict:
    """
    Correct base pace ordering using FP2-derived compound gaps.
    Cross-checks against race gaps.
    Caps at physical maxima when no race data available.
    Falls back to typical values if neither available.
    """
    FALLBACK_SOFT_MEDIUM = 0.4
    FALLBACK_MEDIUM_HARD = 0.3
    MAX_SM_GAP           = 1.0
    MAX_MH_GAP           = 0.5

    corrected  = 0
    fallback   = 0
    overridden = 0

    for circuit, compounds in deg_model.items():
        if 'HARD' not in compounds:
            continue

        hard_base = compounds['HARD']['base_pace']
        fp        = fp_gaps.get(circuit, {})
        rp        = (race_gaps or {}).get(circuit, {})

        # ── MEDIUM vs HARD ───────────────────────────────────
        fp_mh   = fp.get('MEDIUM_vs_HARD')
        race_mh = rp.get('MEDIUM_vs_HARD')

        if fp_mh and race_mh and fp_mh > race_mh * 1.5:
            medium_hard_gap = race_mh
            mh_source       = 'race (FP2 inflated)'
            overridden      += 1
        elif fp_mh and race_mh:
            medium_hard_gap = fp_mh
            mh_source       = 'FP2'
            corrected       += 1
        elif fp_mh and fp_mh <= MAX_MH_GAP:
            medium_hard_gap = fp_mh
            mh_source       = 'FP2'
            corrected       += 1
        elif fp_mh and fp_mh > MAX_MH_GAP:
            medium_hard_gap = MAX_MH_GAP
            mh_source       = 'FP2 capped'
            overridden      += 1
        elif race_mh:
            medium_hard_gap = race_mh
            mh_source       = 'race'
        else:
            medium_hard_gap = FALLBACK_MEDIUM_HARD
            mh_source       = 'fallback'
            fallback        += 1

        # ── SOFT vs MEDIUM ───────────────────────────────────
        fp_sm   = fp.get('SOFT_vs_MEDIUM')
        race_sm = rp.get('SOFT_vs_MEDIUM')

        if fp_sm and race_sm and fp_sm > race_sm * 1.5:
            soft_medium_gap = race_sm
            sm_source       = 'race (FP2 inflated)'
            overridden      += 1
        elif fp_sm and race_sm:
            soft_medium_gap = fp_sm
            sm_source       = 'FP2'
            corrected       += 1
        elif fp_sm and fp_sm <= MAX_SM_GAP:
            soft_medium_gap = fp_sm
            sm_source       = 'FP2'
            corrected       += 1
        elif fp_sm and fp_sm > MAX_SM_GAP:
            soft_medium_gap = MAX_SM_GAP
            sm_source       = 'FP2 capped'
            overridden      += 1
        elif race_sm:
            soft_medium_gap = race_sm
            sm_source       = 'race'
        else:
            soft_medium_gap = FALLBACK_SOFT_MEDIUM
            sm_source       = 'fallback'
            fallback        += 1

        if 'MEDIUM' in compounds:
            compounds['MEDIUM']['base_pace'] = round(
                hard_base - medium_hard_gap, 4)
            compounds['MEDIUM']['pace_source'] = mh_source

        if 'SOFT' in compounds and 'MEDIUM' in compounds:
            compounds['SOFT']['base_pace'] = round(
                compounds['MEDIUM']['base_pace'] -
                soft_medium_gap, 4)
            compounds['SOFT']['pace_source'] = sm_source

    print(f'  Pace gaps: {corrected} FP2, '
          f'{overridden} overridden by race data, '
          f'{fallback} fallback')

    return deg_model


def extract_pit_loss(df: pd.DataFrame) -> dict:
    """Circuit-specific pit lane time loss."""
    known_pit_loss = {
        'Monaco Grand Prix':              27.0,
        'Singapore Grand Prix':           25.0,
        'Hungarian Grand Prix':           24.0,
        'Australian Grand Prix':          24.0,
        'British Grand Prix':             23.0,
        'Spanish Grand Prix':             23.0,
        'Japanese Grand Prix':            23.0,
        'United States Grand Prix':       23.0,
        'Azerbaijan Grand Prix':          23.0,
        'Italian Grand Prix':             22.0,
        'Bahrain Grand Prix':             22.0,
        'Saudi Arabian Grand Prix':       22.0,
        'Abu Dhabi Grand Prix':           22.0,
        'Dutch Grand Prix':               22.0,
        'Chinese Grand Prix':             22.0,
        'Miami Grand Prix':               22.0,
        'Las Vegas Grand Prix':           22.0,
        'Mexico City Grand Prix':         22.0,
        'Brazilian Grand Prix':           22.0,
        'São Paulo Grand Prix':           22.0,
        'Canadian Grand Prix':            22.0,
        'Qatar Grand Prix':               22.0,
        'Emilia Romagna Grand Prix':      22.0,
        'French Grand Prix':              22.0,
        'Belgian Grand Prix':             21.0,
        'Austrian Grand Prix':            21.0,
        'German Grand Prix':              22.0,
        'Russian Grand Prix':             22.0,
        'Portuguese Grand Prix':          22.0,
        'Styrian Grand Prix':             21.0,
        'Tuscan Grand Prix':              22.0,
        'Eifel Grand Prix':               22.0,
        'Sakhir Grand Prix':              19.0,
        '70th Anniversary Grand Prix':    22.0,
    }

    circuits = sorted(df['circuit'].unique())
    pit_loss = {}
    for circuit in circuits:
        pit_loss[circuit] = known_pit_loss.get(circuit, 22.0)
    return pit_loss


def compute_relative_pace(model: dict) -> dict:
    """
    Compute fresh-tyre pace delta between compounds
    per circuit. Positive = first compound is faster.
    """
    relative = {}
    for circuit, compounds in model.items():
        relative[circuit] = {}
        paces = {
            c: compounds[c]['base_pace']
            for c in COMPOUNDS if c in compounds
        }
        if 'SOFT' in paces and 'MEDIUM' in paces:
            relative[circuit]['SOFT_vs_MEDIUM'] = round(
                paces['MEDIUM'] - paces['SOFT'], 3)
        if 'MEDIUM' in paces and 'HARD' in paces:
            relative[circuit]['MEDIUM_vs_HARD'] = round(
                paces['HARD'] - paces['MEDIUM'], 3)
        if 'SOFT' in paces and 'HARD' in paces:
            relative[circuit]['SOFT_vs_HARD'] = round(
                paces['HARD'] - paces['SOFT'], 3)
    return relative


def build_model(save=True) -> dict:
    print('Loading dataset...')
    df = pd.read_csv(DATA_PATH)
    print(f'  {len(df)} stint-laps')
    print(f'  {df["circuit"].nunique()} circuits')

    print('\nFitting degradation curves '
          '(quadratic multiple regression)...')
    deg_model    = fit_degradation_curves(df)
    total_curves = sum(len(v) for v in deg_model.values())
    print(f'  {total_curves} curves fitted across '
          f'{len(deg_model)} circuits')

    print('\nComputing max stint lengths...')
    max_stints = compute_max_stint_lengths(df)

    print('\nApplying Pirelli ratings correction '
          '(sparse circuits only)...')
    deg_model, max_stints = apply_pirelli_ratings_correction(
        deg_model, max_stints)

    print('\nFitting tyre cliff model...')
    cliff_model = fit_cliff_model(df, deg_model)

    print('\nExtracting track temperatures...')
    track_temps = extract_track_temperatures(SEASONS)

    print('\nApplying temperature scaling to cliff model...')
    cliff_model = apply_temperature_scaling(
        cliff_model, deg_model, track_temps)

    print('\nExtracting FP2 compound pace gaps...')
    fp_gaps = extract_fp_pace_gaps(SEASONS)

    print('\nExtracting race pace gaps for validation...')
    race_gaps = extract_race_pace_gaps(df)

    print('\nApplying pace corrections '
          '(FP2 primary, race data validation)...')
    deg_model = apply_fp_pace_gaps(
        deg_model, fp_gaps, race_gaps)

    print('\nExtracting pit loss...')
    pit_loss = extract_pit_loss(df)

    print('\nComputing relative pace...')
    relative = compute_relative_pace(deg_model)

    full_model = {
        'degradation':     deg_model,
        'pit_loss':        pit_loss,
        'relative':        relative,
        'fp_gaps':         fp_gaps,
        'race_gaps':       race_gaps,
        'max_stints':      max_stints,
        'cliff_model':     cliff_model,
        'track_temps':     track_temps,
        'pirelli_ratings': PIRELLI_RATINGS,
        'meta': {
            'seasons':      sorted(
                [int(x) for x in df['year'].unique()]),
            'total_races':  int(
                df[['year', 'round']].drop_duplicates()
                .shape[0]),
            'total_points': int(len(df)),
            'circuits':     sorted(
                df['circuit'].unique().tolist()),
        }
    }

    if save:
        with open(MODEL_PATH, 'w') as f:
            json.dump(full_model, f, indent=2, default=str)
        print(f'\nModel saved to {MODEL_PATH}')

    return full_model


def load_model() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            'Model not found. Run build_model() first.')
    with open(MODEL_PATH) as f:
        return json.load(f)


def print_model_summary(model: dict):
    print('\n=== Model Summary ===')
    print(f'Seasons : {model["meta"]["seasons"]}')
    print(f'Races   : {model["meta"]["total_races"]}')
    print(f'Points  : {model["meta"]["total_points"]}')

    print(f'\nSample degradation curves:')
    for circuit in ['Bahrain Grand Prix',
                    'British Grand Prix',
                    'Japanese Grand Prix']:
        if circuit not in model['degradation']:
            continue
        print(f'\n  {circuit}:')
        for compound, s in model['degradation'][
                circuit].items():
            src   = s.get('pace_source', '?')
            ms    = model['max_stints'].get(
                circuit, {}).get(compound, '?')
            cliff = model.get('cliff_model', {}).get(
                circuit, {}).get(compound, {})
            cliff_str = ''
            if cliff.get('has_cliff'):
                tf = cliff.get('temp_factor', '?')
                cliff_str = (
                    f'  cliff@lap{cliff["cliff_lap"]} '
                    f'({cliff["deg_phase1"]:.3f}→'
                    f'{cliff["deg_phase2"]:.3f}s/lap '
                    f'×{cliff["cliff_ratio"]:.1f} '
                    f'temp×{tf})')
            oc = ' [ordering corrected]' if s.get(
                'ordering_corrected') else ''
            print(f'    {compound:8s} '
                  f'base={s["base_pace"]:.3f}s  '
                  f'deg={s["deg_rate"]:+.4f}s/lap  '
                  f'fuel={s.get("fuel_rate",0):+.4f}s/lap  '
                  f'R²={s["r_squared"]:.3f}  '
                  f'n={s["n_points"]}  '
                  f'max={ms}laps  [{src}]'
                  f'{cliff_str}{oc}')

    print(f'\nRelative pace (fresh tyres):')
    for circuit in ['Bahrain Grand Prix',
                    'British Grand Prix',
                    'Japanese Grand Prix']:
        rel = model['relative'].get(circuit, {})
        if not rel:
            continue
        print(f'  {circuit}:')
        for k, v in rel.items():
            c1, c2 = k.split('_vs_')
            faster = c1 if v > 0 else c2
            slower = c2 if v > 0 else c1
            print(f'    {faster} is {abs(v):.3f}s '
                  f'faster than {slower}')

    print(f'\nPace gap sources:')
    for circuit in ['Bahrain Grand Prix',
                    'Japanese Grand Prix']:
        fp   = model['fp_gaps'].get(circuit, {})
        race = model.get('race_gaps', {}).get(circuit, {})
        print(f'  {circuit}:')
        print(f'    FP2  '
              f'S-M={fp.get("SOFT_vs_MEDIUM","?")}  '
              f'M-H={fp.get("MEDIUM_vs_HARD","?")}')
        print(f'    Race '
              f'S-M={race.get("SOFT_vs_MEDIUM","?")}  '
              f'M-H={race.get("MEDIUM_vs_HARD","?")}')


if __name__ == '__main__':
    print('Building tyre degradation model...\n')
    model = build_model()
    print_model_summary(model)