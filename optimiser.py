"""
optimiser.py — F1 Race Strategy Optimiser
Enumerates all viable 1-stop, 2-stop, and 3-stop strategies
for a given circuit and race distance. Ranks by predicted
total race time. Accounts for tyre regulations (must use
at least 2 different compounds), quadratic degradation,
fuel correction, pit loss, and max stint length constraints.
"""

import json
import itertools
import numpy as np
from model import load_model

COMPOUNDS      = ['SOFT', 'MEDIUM', 'HARD']
MIN_STINT_LAPS = 5


def get_lap_time(circuit: str, compound: str,
                 tyre_age: int, model: dict,
                 race_lap: int = 1) -> float:
    """
    Predict single lap time for a given circuit, compound,
    tyre age and race lap number.
    Includes quadratic degradation and fuel correction.
    Returns None if no curve available.
    """
    deg = model['degradation'].get(circuit, {}).get(compound)
    if deg is None:
        return None

    base       = deg['base_pace']
    deg_rate   = max(0.0, deg['deg_rate'])
    deg_rate_2 = max(0.0, deg.get('deg_rate_2', 0.0))
    fuel_rate  = deg.get('fuel_rate', -0.035)

    return (base
            + deg_rate   * tyre_age
            + deg_rate_2 * tyre_age ** 2
            + fuel_rate  * race_lap)


def stint_time(circuit: str, compound: str,
               start_age: int, laps: int,
               model: dict,
               race_lap_start: int = 1) -> float:
    """
    Total time for a stint of `laps` laps on `compound`
    starting at tyre_age = start_age.

    Uses piecewise degradation model:
    - Phase 1 (stint_lap <= cliff_lap): deg_phase1 s/lap
    - Phase 2 (stint_lap >  cliff_lap): deg_phase2 s/lap

    Falls back to linear model if no cliff detected.
    Includes fuel correction via fuel_rate term.
    Returns None if compound not available for this circuit.
    """
    deg = model['degradation'].get(circuit, {}).get(compound)
    if deg is None:
        return None

    base       = deg['base_pace']
    deg_rate   = max(0.0, deg['deg_rate'])
    deg_rate_2 = max(0.0, deg.get('deg_rate_2', 0.0))
    fuel_rate  = deg.get('fuel_rate', -0.035)

    # Cliff model
    cliff      = model.get('cliff_model', {}).get(
        circuit, {}).get(compound, {})
    has_cliff  = cliff.get('has_cliff', False)
    cliff_lap  = cliff.get('cliff_lap', None)
    deg_phase1 = cliff.get('deg_phase1', deg_rate)
    deg_phase2 = cliff.get('deg_phase2', deg_rate)

    total = 0.0
    for lap in range(laps):
        stint_lap = start_age + lap + 1
        race_lap  = race_lap_start + lap

        if has_cliff and cliff_lap is not None:
            if stint_lap <= cliff_lap:
                effective_deg = deg_phase1 * stint_lap
            else:
                effective_deg = (
                    deg_phase1 * cliff_lap +
                    deg_phase2 * (stint_lap - cliff_lap))
        else:
            effective_deg = (
                deg_rate   * stint_lap +
                deg_rate_2 * stint_lap ** 2)

        lt = base + effective_deg + fuel_rate * race_lap
        total += lt

    return round(total, 3)


def get_available_compounds(circuit: str, model: dict) -> list:
    """Return compounds with fitted curves for this circuit."""
    return [
        c for c in COMPOUNDS
        if c in model['degradation'].get(circuit, {})
    ]


def enumerate_strategies(circuit: str, total_laps: int,
                         model: dict,
                         sc_probability: str = 'low',
                         tyre_sets: dict = None,
                         start_age: int = 0) -> list:
    """
    tyre_sets: {compound: n_sets} or None for unlimited
               e.g. {'SOFT': 2, 'MEDIUM': 3, 'HARD': 2}
    start_age: laps already on starting set (stint 1 only)
    """
    """
    Enumerate all viable strategies and rank by predicted
    total race time.

    sc_probability: 'low', 'medium', 'high'
    Scales effective pit loss to reflect safety car benefit.

    FIA rule: must use at least 2 different dry compounds.
    Max stint length constraints applied per compound per circuit.

    Returns list of strategy dicts sorted by total_time.
    """
    pit_loss_base = model['pit_loss'].get(circuit, 22.0)
    available     = get_available_compounds(circuit, model)
    max_stints    = model.get('max_stints', {}).get(circuit, {})

    if len(available) < 2:
        return []

    sc_factor = {
        'low':    1.0,
        'medium': 0.85,
        'high':   0.70,
    }.get(sc_probability, 1.0)
    effective_pit_loss = pit_loss_base * sc_factor

    def max_lap(compound):
        return max_stints.get(compound, 999)
    
    def check_sets(compounds_used: list) -> bool:
        """Return True if compound usage fits available sets."""
        if not tyre_sets:
            return True
        usage = {}
        for c in compounds_used:
            usage[c] = usage.get(c, 0) + 1
        for c, n in usage.items():
            if c in tyre_sets and n > tyre_sets[c]:
                return False
        return True

    strategies = []

    # ── 1-stop ───────────────────────────────────────────────
    for c1, c2 in itertools.permutations(available, 2):
        for pit1 in range(MIN_STINT_LAPS,
                          total_laps - MIN_STINT_LAPS + 1):
            laps1 = pit1
            laps2 = total_laps - pit1

            if laps2 < MIN_STINT_LAPS:
                continue
            if laps1 > max_lap(c1) or laps2 > max_lap(c2):
                continue
            if not check_sets([c1, c2]):
                continue

            t1 = stint_time(circuit, c1, start_age,
                            laps1, model,
                            race_lap_start=1)
            t2 = stint_time(circuit, c2, 1, laps2, model,
                            race_lap_start=pit1 + 1)

            if t1 is None or t2 is None:
                continue

            total = t1 + t2 + effective_pit_loss

            strategies.append({
                'stops':      1,
                'compounds':  [c1, c2],
                'pit_laps':   [pit1],
                'stint_laps': [laps1, laps2],
                'total_time': round(total, 3),
                'pit_loss':   round(effective_pit_loss, 2),
            })

    # ── 2-stop ───────────────────────────────────────────────
    for compounds in itertools.product(available, repeat=3):
        if len(set(compounds)) < 2:
            continue

        c1, c2, c3 = compounds

        for pit1 in range(MIN_STINT_LAPS,
                          total_laps - 2 * MIN_STINT_LAPS + 1):
            for pit2 in range(pit1 + MIN_STINT_LAPS,
                              total_laps - MIN_STINT_LAPS + 1):
                laps1 = pit1
                laps2 = pit2 - pit1
                laps3 = total_laps - pit2

                if (laps1 < MIN_STINT_LAPS or
                        laps2 < MIN_STINT_LAPS or
                        laps3 < MIN_STINT_LAPS):
                    continue
                if (laps1 > max_lap(c1) or
                        laps2 > max_lap(c2) or
                        laps3 > max_lap(c3)):
                    continue
                if not check_sets([c1, c2, c3]):
                    continue

                t1 = stint_time(circuit, c1, start_age,
                                laps1, model,
                                race_lap_start=1)
                t2 = stint_time(circuit, c2, 1, laps2, model,
                                race_lap_start=pit1 + 1)
                t3 = stint_time(circuit, c3, 1, laps3, model,
                                race_lap_start=pit2 + 1)

                if t1 is None or t2 is None or t3 is None:
                    continue

                total = (t1 + t2 + t3 +
                         2 * effective_pit_loss)

                strategies.append({
                    'stops':      2,
                    'compounds':  list(compounds),
                    'pit_laps':   [pit1, pit2],
                    'stint_laps': [laps1, laps2, laps3],
                    'total_time': round(total, 3),
                    'pit_loss':   round(effective_pit_loss, 2),
                })

    # ── 3-stop ───────────────────────────────────────────────
    for compounds in itertools.product(available, repeat=4):
        if len(set(compounds)) < 2:
            continue

        c1, c2, c3, c4 = compounds

        base_stint = total_laps // 4
        for offset in range(-3, 4):
            pit1 = base_stint + offset
            pit2 = 2 * base_stint + offset
            pit3 = 3 * base_stint + offset

            if (pit1 < MIN_STINT_LAPS or
                    pit2 <= pit1 + MIN_STINT_LAPS or
                    pit3 <= pit2 + MIN_STINT_LAPS or
                    total_laps - pit3 < MIN_STINT_LAPS):
                continue

            laps1 = pit1
            laps2 = pit2 - pit1
            laps3 = pit3 - pit2
            laps4 = total_laps - pit3

            if (laps1 > max_lap(c1) or
                    laps2 > max_lap(c2) or
                    laps3 > max_lap(c3) or
                    laps4 > max_lap(c4)):
                continue
            if not check_sets([c1, c2, c3, c4]):
                continue

            t1 = stint_time(circuit, c1, start_age,
                            laps1, model,
                            race_lap_start=1)
            t2 = stint_time(circuit, c2, 1, laps2, model,
                            race_lap_start=pit1 + 1)
            t3 = stint_time(circuit, c3, 1, laps3, model,
                            race_lap_start=pit2 + 1)
            t4 = stint_time(circuit, c4, 1, laps4, model,
                            race_lap_start=pit3 + 1)

            if (t1 is None or t2 is None or
                    t3 is None or t4 is None):
                continue

            total = (t1 + t2 + t3 + t4 +
                     3 * effective_pit_loss)

            strategies.append({
                'stops':      3,
                'compounds':  list(compounds),
                'pit_laps':   [pit1, pit2, pit3],
                'stint_laps': [laps1, laps2, laps3, laps4],
                'total_time': round(total, 3),
                'pit_loss':   round(effective_pit_loss, 2),
            })

    if not strategies:
        return []

    strategies.sort(key=lambda x: x['total_time'])

    best_time = strategies[0]['total_time']
    for s in strategies:
        s['delta_vs_optimal'] = round(
            s['total_time'] - best_time, 3)

    # Deduplicate — keep best per (stops, compound sequence)
    seen   = set()
    unique = []
    for s in strategies:
        key = (s['stops'], tuple(s['compounds']))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def compute_pit_windows(strategies: list,
                        circuit: str,
                        total_laps: int,
                        model: dict,
                        sc_probability: str = 'low',
                        threshold: float = 1.5) -> list:
    """
    For each unique compound sequence in the top strategies,
    find the pit window — the range of pit laps that produce
    a total race time within threshold seconds of the optimal
    for that compound sequence.

    Returns the top strategies with pit_windows added:
    pit_windows: list of [earliest_pit, latest_pit] per stop
    """
    pit_loss_base = model['pit_loss'].get(circuit, 22.0)
    max_stints    = model.get('max_stints', {}).get(circuit, {})

    sc_factor = {
        'low':    1.0,
        'medium': 0.85,
        'high':   0.70,
    }.get(sc_probability, 1.0)
    effective_pit_loss = pit_loss_base * sc_factor

    def max_lap(compound):
        return max_stints.get(compound, 999)

    result = []

    for s in strategies:
        compounds  = s['compounds']
        stops      = s['stops']
        best_time  = s['total_time']

        # Find all pit lap combinations for this compound
        # sequence within threshold of its own best time
        windows = []

        if stops == 1:
            c1, c2 = compounds
            pit1_times = {}
            for pit1 in range(5, total_laps - 5):
                laps1 = pit1
                laps2 = total_laps - pit1
                if laps1 < 5 or laps2 < 5:
                    continue
                if laps1 > max_lap(c1) or laps2 > max_lap(c2):
                    continue
                t1 = stint_time(circuit, c1, 1, laps1, model,
                                race_lap_start=1)
                t2 = stint_time(circuit, c2, 1, laps2, model,
                                race_lap_start=pit1 + 1)
                if t1 is None or t2 is None:
                    continue
                total = t1 + t2 + effective_pit_loss
                pit1_times[pit1] = total

            if pit1_times:
                best = min(pit1_times.values())
                valid = [p for p, t in pit1_times.items()
                         if t <= best + threshold]
                if valid:
                    windows = [[min(valid), max(valid)]]

        elif stops == 2:
            c1, c2, c3 = compounds
            # Fix pit2 relative to pit1 at optimal ratio
            # then vary pit1 to find window
            opt_pit1 = s['pit_laps'][0]
            opt_pit2 = s['pit_laps'][1]
            gap      = opt_pit2 - opt_pit1

            pit1_times = {}
            for pit1 in range(5, total_laps - 10):
                pit2  = pit1 + gap
                laps1 = pit1
                laps2 = pit2 - pit1
                laps3 = total_laps - pit2
                if (laps1 < 5 or laps2 < 5 or
                        laps3 < 5 or pit2 >= total_laps):
                    continue
                if (laps1 > max_lap(c1) or
                        laps2 > max_lap(c2) or
                        laps3 > max_lap(c3)):
                    continue
                t1 = stint_time(circuit, c1, 1, laps1, model,
                                race_lap_start=1)
                t2 = stint_time(circuit, c2, 1, laps2, model,
                                race_lap_start=pit1 + 1)
                t3 = stint_time(circuit, c3, 1, laps3, model,
                                race_lap_start=pit2 + 1)
                if t1 is None or t2 is None or t3 is None:
                    continue
                total = t1 + t2 + t3 + 2 * effective_pit_loss
                pit1_times[pit1] = total

            if pit1_times:
                best = min(pit1_times.values())
                valid = [p for p, t in pit1_times.items()
                         if t <= best + threshold]
                if valid:
                    windows = [
                        [min(valid), max(valid)],
                        [min(valid) + gap, max(valid) + gap]
                    ]

        elif stops == 3:
            # Use fixed gaps from optimal, vary first pit
            opt_pit1 = s['pit_laps'][0]
            opt_pit2 = s['pit_laps'][1]
            opt_pit3 = s['pit_laps'][2]
            gap1     = opt_pit2 - opt_pit1
            gap2     = opt_pit3 - opt_pit2
            c1, c2, c3, c4 = compounds

            pit1_times = {}
            for pit1 in range(5, total_laps - 15):
                pit2  = pit1 + gap1
                pit3  = pit2 + gap2
                laps1 = pit1
                laps2 = gap1
                laps3 = gap2
                laps4 = total_laps - pit3
                if (laps1 < 5 or laps2 < 5 or
                        laps3 < 5 or laps4 < 5 or
                        pit3 >= total_laps):
                    continue
                if (laps1 > max_lap(c1) or
                        laps2 > max_lap(c2) or
                        laps3 > max_lap(c3) or
                        laps4 > max_lap(c4)):
                    continue
                t1 = stint_time(circuit, c1, 1, laps1, model,
                                race_lap_start=1)
                t2 = stint_time(circuit, c2, 1, laps2, model,
                                race_lap_start=pit1 + 1)
                t3 = stint_time(circuit, c3, 1, laps3, model,
                                race_lap_start=pit2 + 1)
                t4 = stint_time(circuit, c4, 1, laps4, model,
                                race_lap_start=pit3 + 1)
                if (t1 is None or t2 is None or
                        t3 is None or t4 is None):
                    continue
                total = (t1 + t2 + t3 + t4 +
                         3 * effective_pit_loss)
                pit1_times[pit1] = total

            if pit1_times:
                best = min(pit1_times.values())
                valid = [p for p, t in pit1_times.items()
                         if t <= best + threshold]
                if valid:
                    windows = [
                        [min(valid), max(valid)],
                        [min(valid) + gap1, max(valid) + gap1],
                        [min(valid) + gap1 + gap2,
                         max(valid) + gap1 + gap2],
                    ]

        s_copy = dict(s)
        s_copy['pit_windows'] = windows
        result.append(s_copy)

    return result

def format_strategy(s: dict, model: dict,
                    circuit: str) -> str:
    """Format a strategy for terminal display."""
    compounds  = s['compounds']
    pit_laps   = s['pit_laps']
    stint_laps = s['stint_laps']
    delta      = s['delta_vs_optimal']

    compound_str = ' → '.join(compounds)
    pit_str      = ', '.join([f'Lap {p}' for p in pit_laps])

    mins = int(s['total_time'] // 60)
    secs = s['total_time'] % 60

    delta_str = (
        'OPTIMAL' if delta == 0
        else f'+{delta:.3f}s'
    )

    lines = [
        f"{s['stops']}-stop: {compound_str}",
        f"  Pit: {pit_str}",
        f"  Stints: {stint_laps}",
        f"  Total: {mins}m {secs:.3f}s  ({delta_str})",
    ]

    for i, (c, laps) in enumerate(
            zip(compounds, stint_laps)):
        deg = model['degradation'].get(
            circuit, {}).get(c, {})
        if deg:
            deg_rate  = max(0, deg['deg_rate'])
            deg_rate_2 = max(0, deg.get('deg_rate_2', 0))
            deg_total = (deg_rate * (laps - 1) +
                         deg_rate_2 * (laps - 1) ** 2)
            lines.append(
                f'  Stint {i+1} {c} {laps} laps: '
                f'+{deg_total:.3f}s/lap pace loss by end '
                f'({deg_rate:.3f}s/lap deg rate)')

    return '\n'.join(lines)


def optimise(circuit: str, total_laps: int,
             sc_probability: str = 'low',
             top_n: int = 15,
             tyre_sets: dict = None,
             start_age: int = 0) -> list:
    model      = load_model()
    strategies = enumerate_strategies(
        circuit, total_laps, model, sc_probability,
        tyre_sets=tyre_sets, start_age=start_age)
    top        = strategies[:top_n]
    top        = compute_pit_windows(
        top, circuit, total_laps, model, sc_probability)
    return top

if __name__ == '__main__':
    model   = load_model()
    circuit = 'Japanese Grand Prix'
    laps    = 53

    print(f'Optimising strategy for {circuit} ({laps} laps)\n')

    available  = get_available_compounds(circuit, model)
    max_stints = model.get('max_stints', {}).get(circuit, {})
    print(f'Available compounds: {available}')
    print(f'Max stint lengths: {max_stints}')
    print(f'Pit loss: {model["pit_loss"].get(circuit, 22.0)}s\n')

    for sc in ['low', 'medium', 'high']:
        print(f'── Safety car probability: {sc.upper()} ──\n')
        strategies = enumerate_strategies(
            circuit, laps, model, sc)[:5]

        if not strategies:
            print('  No strategies found\n')
            continue

        for i, s in enumerate(strategies):
            print(f'#{i+1} ' + format_strategy(
                s, model, circuit))
            print()