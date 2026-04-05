# F1 Race Strategy Optimiser

A race strategy optimisation tool trained on 112,732 stint-laps across 
2020–2025. Predicts optimal pit stop strategies by modelling tyre 
degradation, fuel burn, compound pace gaps and safety car probability.

**Live tool:** https://web-production-9f341.up.railway.app/

---

## What it does

Given a circuit, race distance and safety car probability, the tool 
enumerates all viable 1-stop, 2-stop and 3-stop strategies and ranks 
them by predicted total race time. For each strategy it computes a 
**pit window** — the range of lap numbers within 1.5s of optimal — 
matching the format used by Pirelli in their pre-race strategy graphics.

Users can specify tyre availability (sets per compound, age of each set) 
and which set they are starting on. Safety car probability scales the 
effective pit loss by 1.0× / 0.85× / 0.70×.

---

## Model architecture

### 1. Degradation curves

Fitted per compound per circuit using **quadratic multiple regression**:
lap_time = base + fuel_rate × race_lap
+ deg_rate  × stint_lap
+ deg_rate² × stint_lap²

Separating `fuel_rate` and `deg_rate` in a single regression avoids the 
collinearity error introduced by sequential correction methods. Physical 
bounds enforced: fuel rate constrained to [−0.150, −0.010] s/lap; 
degradation ordering SOFT ≥ MEDIUM ≥ HARD enforced across all circuits.

### 2. Piecewise tyre cliff model

A two-segment piecewise linear model is fitted per compound per circuit 
to capture the tyre performance cliff:

- **Phase 1** (stint lap ≤ cliff lap): grip-dominated linear wear
- **Phase 2** (stint lap > cliff lap): thermally-driven accelerated wear

Breakpoint identified by minimising RSS across candidate laps 5–25. 
cliff_ratio = phase2 / phase1 ≥ 1.5 required for cliff to be applied.

At high-stress circuits with sparse SOFT data (n < 300 points), a 
physics-informed prior is applied: cliff at lap 8, phase2 = phase1 × 2.5. 
Triggered when Pirelli lateral or tyre_stress rating ≥ 4.

Cliff phase2 degradation rate scaled by track temperature via an 
Arrhenius relationship (reference 35°C, SOFT sensitivity k=0.15).

### 3. Compound pace gaps

Base pace ordering corrected using FP2 long-run data extracted from 
109 sessions across 2020–2025. Validation pipeline:

1. Extract FP2 10th-percentile pace per compound per session
2. Extract race 10th-percentile fuel-corrected pace per compound
3. If FP2 gap > race gap × 1.5 → override with race data
4. Physical caps: S-M ≤ 1.0s, M-H ≤ 0.5s
5. Fallback: S-M = 0.4s, M-H = 0.3s

### 4. Pirelli circuit ratings

Pirelli's published circuit characterisation ratings (tyre stress, 
lateral forces, abrasion, traction, braking, track evolution, downforce 
— each 1–5) are used as a fallback correction for data-sparse circuits 
(n < 300 points or < 4 dry races). Applied additively to deg_rate; 
well-data circuits are unaffected.

### 5. Max stint length constraints

Maximum observed stint length per compound per circuit derived from 
data with 5% buffer. Prevents extrapolation beyond observed tyre life. 
Scaled by Pirelli tyre_stress rating for sparse circuits.

### 6. Optimiser

Exhaustive enumeration of 1-stop, 2-stop and 3-stop strategies. Each 
strategy evaluated via lap-by-lap simulation using piecewise degradation 
model with fuel correction. FIA compound rule enforced (≥ 2 different 
compounds). Strategies ranked by total race time.

---

## Validation

### Fuel correction

| Metric | Model | Literature | Assessment |
|--------|-------|------------|------------|
| Mean fuel rate | −0.037 s/kg | 0.028–0.040 s/kg | ✓ Within range |
| Direction | All negative | Must be negative | ✓ Correct |
| Circuit ordering | Qatar highest, Monza lowest | Physically expected | ✓ Correct |

Reference: Royal Academy of Engineering (2010), *Formula One Race Strategy*.

### Degradation rates

Spearman rank correlation between model-fitted SOFT degradation rates 
and Pirelli circuit severity ratings across 29 circuits:

| Pirelli Rating | Spearman r | p-value | Significance |
|----------------|-----------|---------|--------------|
| Abrasion | 0.502 | 0.006 | ** (99%) |
| Tyre stress | 0.434 | 0.019 | * (95%) |
| Lateral forces | 0.391 | 0.036 | * (95%) |

All correlations positive and statistically significant (p < 0.05).

Circuit-level validation:

| Circuit | SOFT deg (model) | Pirelli stress | Assessment |
|---------|-----------------|----------------|------------|
| Qatar GP | 0.167 s/lap | 5/5 | ✓ Highest |
| Japanese GP | 0.121 s/lap | 5/5 | ✓ Very high |
| Spanish GP | 0.098 s/lap | 4/5 | ✓ High |
| Bahrain GP | 0.088 s/lap | 4/5 | ✓ Moderate-high |
| Italian GP | 0.005 s/lap | 3/5 lateral=2 | ✓ Lowest |
| Hungarian GP | 0.020 s/lap | 3/5 abrasion=2 | ✓ Low |


### R² distribution

| Compound | Mean R² | % circuits R² > 0.3 |
|----------|---------|---------------------|
| SOFT | 0.409 | 72.4% |
| MEDIUM | 0.268 | 43.8% |
| HARD | 0.296 | 34.4% |

Low R² is expected — race lap time variance is dominated by traffic, 
safety cars and driver management decisions outside the model scope. 
The model isolates the marginal degradation effect, not absolute lap time.

---

## Known limitations

**Single-car model.** No competitor strategy, traffic or 
undercut/overcut dynamics modelled.

**Tyre temperature management.** At high-lateral circuits (Suzuka, 
Zandvoort), the model may underweight MEDIUM→HARD strategies. Tyre 
surface temperature management by drivers is invisible in public lap 
time data — this requires Pirelli's internal tyre temperature sensors.

**Pit loss hardcoded.** Circuit-specific pit loss from published 
estimates, not fitted from data.

**Low R².** Expected given noise sources but limits precision of 
individual lap time predictions.

**2026 regulations not modelled.** Tool calibrated on 2020–2025 data 
under current technical regulations.

---

## Data

- **Source:** FastF1 API
- **Coverage:** 125/127 races across 2020–2025 (98.4%)
- **Volume:** 112,732 clean dry stint-laps
- **Circuits:** 32
- **Filters:** Wet laps, pit in/out laps, safety car laps 
  (TrackStatus ≠ '1'), deleted laps, outliers (>15% slower than fastest)
- **Fuel correction:** 0.03 s/kg, 110 kg start, 1.6 kg/lap burn rate

---

## Stack

- Python, Flask, FastF1, NumPy, pandas, SciPy
- Deployed on Railway

---

## Author

Mahdi Kadiri   
MEng Mechanical Engineering (Automotive), University of Bath  
Targeting F1 vehicle dynamics and performance engineering placements