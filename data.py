"""
data.py — F1 Stint Data Pipeline
Pulls race stint data from FastF1 API for the specified seasons.
Extracts clean dry stint-laps with fuel correction.
Also extracts track temperatures from race weather data.
Saves to data/stints.csv and data/track_temps.json.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import fastf1

os.makedirs('cache', exist_ok=True)
os.makedirs('data', exist_ok=True)
fastf1.Cache.enable_cache('cache')

SEASONS       = list(range(2020, 2026))
DATA_PATH     = 'data/stints.csv'
META_PATH     = 'data/race_metadata.csv'
TEMPS_PATH    = 'data/track_temps.json'

DRY_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD'}

FUEL_LOAD_KG       = 110.0
FUEL_BURN_KG_LAP   = 1.6
FUEL_TIME_LOSS_S   = 0.03


def fuel_corrected_lap_time(lap_time: float,
                             race_lap: int) -> float:
    """
    Correct lap time for fuel load effect.
    Car gets lighter as fuel burns off — removes ~0.03s/kg.
    """
    fuel_remaining = FUEL_LOAD_KG - (race_lap * FUEL_BURN_KG_LAP)
    fuel_remaining = max(0, fuel_remaining)
    fuel_correction = (FUEL_LOAD_KG - fuel_remaining) * FUEL_TIME_LOSS_S
    return lap_time - fuel_correction


def is_wet_race(laps: pd.DataFrame,
                threshold: float = 0.30) -> bool:
    """
    Flag race as wet if more than threshold fraction of laps
    are on WET or INTERMEDIATE compound.
    """
    wet_compounds = {'WET', 'INTERMEDIATE'}
    wet_laps = laps['Compound'].isin(wet_compounds).sum()
    return (wet_laps / len(laps)) > threshold


def extract_race_stints(session) -> list:
    """
    Extract clean dry stint-laps from a race session.

    Filters:
    - Wet laps (non-dry compound)
    - Pit in/out laps
    - Safety car laps (TrackStatus != '1')
    - Deleted laps
    - Outliers (>15% slower than session fastest)
    - Laps with missing data

    Returns list of dicts, one per lap.
    """
    circuit   = session.event['EventName']
    year      = session.event['year']
    round_num = session.event['RoundNumber']
    total_laps = session.total_laps

    all_laps = session.laps.copy()

    if all_laps.empty:
        return []

    wet_race = is_wet_race(all_laps)

    # Deleted laps mask
    deleted_mask = (all_laps['Deleted']
                    .fillna(False)
                    .infer_objects(copy=False))

    # Clean lap filter
    clean = all_laps[
        all_laps['Compound'].isin(DRY_COMPOUNDS) &
        all_laps['PitInTime'].isna() &
        all_laps['PitOutTime'].isna() &
        all_laps['LapTime'].notna() &
        all_laps['TrackStatus'].eq('1') &
        deleted_mask.eq(False)
    ].copy()

    if clean.empty:
        return []

    clean['LapTimeSec'] = clean['LapTime'].dt.total_seconds()

    fastest = clean['LapTimeSec'].min()
    clean   = clean[clean['LapTimeSec'] < fastest * 1.15]

    if clean.empty:
        return []

    rows = []
    for _, lap in clean.iterrows():
        stint_lap = int(lap['TyreLife']) if pd.notna(
            lap['TyreLife']) else None
        tyre_age  = int(lap['TyreLife']) if pd.notna(
            lap['TyreLife']) else None
        race_lap  = int(lap['LapNumber']) if pd.notna(
            lap['LapNumber']) else None

        if stint_lap is None or race_lap is None:
            continue
        if stint_lap < 1 or race_lap < 1:
            continue

        lap_time_raw     = float(lap['LapTimeSec'])
        lap_time_fuel    = fuel_corrected_lap_time(
            lap_time_raw, race_lap)
        wet_lap_count    = int(all_laps[
            all_laps['Compound'].isin(
                {'WET', 'INTERMEDIATE'})].shape[0])

        rows.append({
            'year':              int(year),
            'circuit':           circuit,
            'round':             int(round_num),
            'driver':            str(lap['Driver']),
            'compound':          str(lap['Compound']),
            'stint_lap':         stint_lap,
            'tyre_age':          tyre_age,
            'race_lap':          race_lap,
            'lap_time_raw':      round(lap_time_raw, 4),
            'lap_time_fuel_adj': round(lap_time_fuel, 4),
            'total_race_laps':   int(total_laps),
            'wet_race':          bool(wet_race),
            'wet_lap_count':     wet_lap_count,
        })

    return rows


def extract_track_temperatures(
        seasons: list,
        save_path: str = TEMPS_PATH) -> dict:
    """
    Extract average race-time track temperature per circuit
    from FastF1 weather data.

    Uses race sessions only. Averages middle third of race
    to avoid formation lap and cool-down lap outliers.
    Averages across all available seasons per circuit.

    Returns:
    {circuit: {'track_temp': float, 'air_temp': float,
               'n_races': int}}
    """
    if os.path.exists(save_path):
        print('  Loading cached track temperatures...')
        with open(save_path) as f:
            return json.load(f)

    temps_by_circuit = {}
    session_count    = 0

    for year in seasons:
        print(f'  Extracting temperatures for {year}...')
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
                ses = fastf1.get_session(year, round_num, 'R')
                ses.load(laps=False, telemetry=False,
                         weather=True, messages=False)
            except Exception:
                time.sleep(2)
                continue

            try:
                circuit = ses.event['EventName']
                weather = ses.weather_data

                if weather is None or weather.empty:
                    time.sleep(2)
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
                print(f'    {circuit} {year}: '
                      f'{track_temp:.1f}°C track')

            except Exception as e:
                print(f'    Error: {e}')

            time.sleep(3)

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

    print(f'  Track temperatures extracted: '
          f'{session_count} races, '
          f'{len(result)} circuits')

    return result


def build_dataset(seasons: list,
                  save_path: str = DATA_PATH,
                  meta_path: str = META_PATH,
                  checkpoint_every: int = 5) -> pd.DataFrame:
    """
    Build the full stint dataset from FastF1.
    Resumes from existing data if available.
    Saves checkpoint every N races.
    """
    existing_df  = None
    done_races   = set()
    all_rows     = []
    race_meta    = []

    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        print(f'  Resuming — {len(existing_df)} rows loaded')
        for _, g in existing_df.groupby(['year', 'round']):
            done_races.add(
                (int(g['year'].iloc[0]),
                 int(g['round'].iloc[0])))
        all_rows = existing_df.to_dict('records')

    race_count = 0

    for year in seasons:
        print(f'\nProcessing {year}...')
        try:
            schedule = fastf1.get_event_schedule(
                year, include_testing=False)
        except Exception as e:
            print(f'  Failed to get schedule: {e}')
            continue

        for _, event in schedule.iterrows():
            round_num = int(event['RoundNumber'])
            if round_num == 0:
                continue

            key = (year, round_num)
            if key in done_races:
                print(f'  Skipping {event["EventName"]} '
                      f'{year} (already done)')
                continue

            print(f'  Loading {event["EventName"]} {year}...')

            try:
                ses = fastf1.get_session(year, round_num, 'R')
                ses.load(laps=True, telemetry=False,
                         weather=False, messages=False)
            except Exception as e:
                print(f'    Failed to load: {e}')
                time.sleep(5)
                continue

            rows = extract_race_stints(ses)

            if rows:
                all_rows.extend(rows)
                race_meta.append({
                    'year':     year,
                    'round':    round_num,
                    'circuit':  event['EventName'],
                    'n_laps':   len(rows),
                })
                print(f'    {len(rows)} stint-laps extracted')
            else:
                print(f'    No clean laps found')

            done_races.add(key)
            race_count += 1

            if race_count % checkpoint_every == 0:
                df = pd.DataFrame(all_rows).drop_duplicates()
                df.to_csv(save_path, index=False)
                print(f'    Checkpoint saved '
                      f'({len(df)} total rows)')

            time.sleep(5)

    df = pd.DataFrame(all_rows).drop_duplicates()
    df.to_csv(save_path, index=False)
    print(f'\nDataset saved: {len(df)} rows, '
          f'{df["circuit"].nunique()} circuits')

    if race_meta:
        meta_df = pd.DataFrame(race_meta)
        meta_df.to_csv(meta_path, index=False)

    return df


if __name__ == '__main__':
    print('Building F1 stint dataset...\n')

    print('Extracting stint data...')
    df = build_dataset(SEASONS)

    print('\nExtracting track temperatures...')
    temps = extract_track_temperatures(SEASONS)

    print(f'\nDone.')
    print(f'  Stints: {len(df)} rows, '
          f'{df["circuit"].nunique()} circuits')
    print(f'  Temperatures: {len(temps)} circuits')

    print('\nSample temperatures:')
    for c in ['Bahrain Grand Prix',
              'Japanese Grand Prix',
              'British Grand Prix',
              'Qatar Grand Prix',
              'Monaco Grand Prix']:
        t = temps.get(c)
        if t:
            print(f'  {c}: {t["track_temp"]}°C track, '
                  f'{t["air_temp"]}°C air '
                  f'(n={t["n_races"]} races)')