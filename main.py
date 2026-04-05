"""
main.py — F1 Strategy Optimiser Web App
"""

import os
import json
import fastf1
from flask import Flask, jsonify, request
from optimiser import optimise, get_available_compounds
from model import load_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

app = Flask(__name__)

COMPOUND_COLOURS = {
    'SOFT':   '#e8002d',
    'MEDIUM': '#ffd700',
    'HARD':   '#ffffff',
}

LAUNCHER_HTML = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>F1 Strategy Optimiser</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0a0a0a; color: #fff;
  font-family: 'Segoe UI', sans-serif;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  min-height: 100vh; padding: 40px;
}
h1 {
  font-size: 28px; font-weight: 300;
  letter-spacing: 4px; text-transform: uppercase;
  margin-bottom: 6px;
}
.subtitle {
  font-size: 12px; color: #888; letter-spacing: 2px;
  text-transform: uppercase; margin-bottom: 48px;
}
.card {
  background: #0f0f0f; border: 1px solid #1e1e1e;
  border-radius: 12px; padding: 40px; width: 100%;
  max-width: 680px;
}
.row {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; margin-bottom: 16px;
}
.row.full { grid-template-columns: 1fr; }
.field { display: flex; flex-direction: column; gap: 6px; }
label {
  font-size: 10px; color: #aaa;
  text-transform: uppercase; letter-spacing: 1.5px;
}
select, input[type="number"] {
  background: #161616; border: 1px solid #2a2a2a;
  color: #fff; padding: 10px 14px; border-radius: 6px;
  font-size: 13px; font-family: 'Segoe UI', sans-serif;
  appearance: none; width: 100%;
  transition: border-color 0.2s;
}
select:hover, input:hover { border-color: #555; }
select:focus, input:focus {
  outline: none; border-color: #777;
}
.btn-run {
  width: 100%; padding: 14px; border-radius: 6px;
  background: #e8002d; border: none; color: #fff;
  font-size: 13px; font-weight: 600; cursor: pointer;
  letter-spacing: 1px; text-transform: uppercase;
  transition: all 0.2s; margin-top: 8px;
}
.btn-run:hover:not(:disabled) {
  background: #ff0033; transform: translateY(-1px);
}
.btn-run:disabled {
  opacity: 0.4; cursor: not-allowed; transform: none;
}
.divider {
  border: none; border-top: 1px solid #1a1a1a;
  margin: 24px 0;
}
.status-box {
  margin-top: 24px; background: #080808;
  border: 1px solid #1a1a1a; border-radius: 6px;
  padding: 16px; min-height: 60px;
  font-size: 11px; font-family: monospace; display: none;
}
.status-line { color: #888; margin-bottom: 4px; }
.status-line.active { color: #4a9eff; }
.status-line.done   { color: #22cc55; }
.status-line.error  { color: #e8002d; }
.sc-row {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
}
.sc-btn {
  padding: 10px; border-radius: 6px; cursor: pointer;
  font-size: 11px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; border: 1px solid #2a2a2a;
  background: #161616; color: #888;
  transition: all 0.2s; text-align: center;
}
.sc-btn.selected-low {
  background:#1a2a1a; color:#22cc55;
  border-color:#22cc55;
}
.sc-btn.selected-medium {
  background:#2a2a1a; color:#ffd700;
  border-color:#ffd700;
}
.sc-btn.selected-high {
  background:#2a1a1a; color:#e8002d;
  border-color:#e8002d;
}
.section-label {
  font-size: 10px; color: #aaa; text-transform: uppercase;
  letter-spacing: 1.5px; margin-bottom: 12px;
  margin-top: 4px;
}
.tyre-block {
  background: #111; border: 1px solid #1e1e1e;
  border-radius: 8px; padding: 16px; margin-bottom: 10px;
}
.tyre-header {
  display: flex; align-items: center;
  justify-content: space-between; margin-bottom: 12px;
}
.tyre-label {
  font-size: 12px; font-weight: 700;
  letter-spacing: 1px; padding: 3px 10px;
  border-radius: 3px;
}
.tyre-label.soft   { background:#e8002d; color:#fff; }
.tyre-label.medium { background:#ffd700; color:#000; }
.tyre-label.hard   { background:#aaa;    color:#000; }
.toggle-row { display: flex; gap: 8px; }
.toggle-btn {
  padding: 6px 14px; border-radius: 4px; cursor: pointer;
  font-size: 10px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; border: 1px solid #2a2a2a;
  background: #161616; color: #888; transition: all 0.2s;
}
.toggle-btn.active {
  background: #1a2a1a; color: #22cc55;
  border-color: #22cc55;
}
.tyre-inputs { display: none; margin-top: 12px; }
.tyre-inputs.visible { display: block; }
.sets-row {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 10px;
}
.sets-row label {
  font-size: 10px; color: #aaa;
  text-transform: uppercase; letter-spacing: 1px;
  white-space: nowrap;
}
.sets-row input { width: 70px; }
.ages-row {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-top: 8px;
}
.age-input-wrap {
  display: flex; flex-direction: column; gap: 4px;
}
.age-input-wrap label { font-size: 9px; color: #aaa; }
.age-input-wrap input { width: 65px; }
.start-btns {
  display: flex; flex-wrap: wrap; gap: 6px;
}
.start-btn {
  padding: 6px 12px; border-radius: 4px;
  cursor: pointer; font-size: 10px;
  font-weight: 600; letter-spacing: 1px;
  border: 1px solid #2a2a2a;
  background: #161616; color: #888;
  transition: all 0.2s; white-space: nowrap;
}
.start-btn.selected {
  background: #1a2a1a; color: #22cc55;
  border-color: #22cc55;
}
</style>
</head>
<body>
<h1>F1 Strategy</h1>
<p class="subtitle">Race Strategy Optimiser</p>

<div class="card">
  <div class="row">
    <div class="field">
      <label>Circuit</label>
      <select id="circuit"></select>
    </div>
    <div class="field">
      <label>Race Laps</label>
      <input type="number" id="laps"
             value="57" min="20" max="80">
    </div>
  </div>

  <div class="row full">
    <div class="field">
      <label>Safety Car Probability</label>
      <div class="sc-row">
        <div class="sc-btn selected-low" id="sc-low"
             onclick="selectSC('low')">Low</div>
        <div class="sc-btn" id="sc-medium"
             onclick="selectSC('medium')">Medium</div>
        <div class="sc-btn" id="sc-high"
             onclick="selectSC('high')">High</div>
      </div>
    </div>
  </div>

  <hr class="divider">
  <div class="section-label">Tyre Availability</div>

  <!-- SOFT -->
  <div class="tyre-block">
    <div class="tyre-header">
      <span class="tyre-label soft">SOFT</span>
      <div class="toggle-row">
        <div class="toggle-btn active"
             id="toggle-soft-unlimited"
             onclick="setMode('soft','unlimited')">
          Unlimited</div>
        <div class="toggle-btn"
             id="toggle-soft-specify"
             onclick="setMode('soft','specify')">
          Specify</div>
      </div>
    </div>
    <div class="tyre-inputs" id="inputs-soft">
      <div class="sets-row">
        <label>Sets available</label>
        <input type="number" id="sets-soft"
               value="8" min="0" max="12"
               oninput="updateAgeInputs('soft')">
      </div>
      <div class="ages-row" id="ages-soft"></div>
      <div class="start-row" id="start-row-soft"
           style="display:none;margin-top:12px;">
        <label style="font-size:10px;color:#aaa;
               text-transform:uppercase;letter-spacing:1px;
               margin-bottom:6px;display:block;">
          Starting on</label>
        <div class="start-btns"
             id="start-btns-soft"></div>
      </div>
    </div>
  </div>

  <!-- MEDIUM -->
  <div class="tyre-block">
    <div class="tyre-header">
      <span class="tyre-label medium">MEDIUM</span>
      <div class="toggle-row">
        <div class="toggle-btn active"
             id="toggle-medium-unlimited"
             onclick="setMode('medium','unlimited')">
          Unlimited</div>
        <div class="toggle-btn"
             id="toggle-medium-specify"
             onclick="setMode('medium','specify')">
          Specify</div>
      </div>
    </div>
    <div class="tyre-inputs" id="inputs-medium">
      <div class="sets-row">
        <label>Sets available</label>
        <input type="number" id="sets-medium"
               value="3" min="0" max="8"
               oninput="updateAgeInputs('medium')">
      </div>
      <div class="ages-row" id="ages-medium"></div>
      <div class="start-row" id="start-row-medium"
           style="display:none;margin-top:12px;">
        <label style="font-size:10px;color:#aaa;
               text-transform:uppercase;letter-spacing:1px;
               margin-bottom:6px;display:block;">
          Starting on</label>
        <div class="start-btns"
             id="start-btns-medium"></div>
      </div>
    </div>
  </div>

  <!-- HARD -->
  <div class="tyre-block">
    <div class="tyre-header">
      <span class="tyre-label hard">HARD</span>
      <div class="toggle-row">
        <div class="toggle-btn active"
             id="toggle-hard-unlimited"
             onclick="setMode('hard','unlimited')">
          Unlimited</div>
        <div class="toggle-btn"
             id="toggle-hard-specify"
             onclick="setMode('hard','specify')">
          Specify</div>
      </div>
    </div>
    <div class="tyre-inputs" id="inputs-hard">
      <div class="sets-row">
        <label>Sets available</label>
        <input type="number" id="sets-hard"
               value="2" min="0" max="6"
               oninput="updateAgeInputs('hard')">
      </div>
      <div class="ages-row" id="ages-hard"></div>
      <div class="start-row" id="start-row-hard"
           style="display:none;margin-top:12px;">
        <label style="font-size:10px;color:#aaa;
               text-transform:uppercase;letter-spacing:1px;
               margin-bottom:6px;display:block;">
          Starting on</label>
        <div class="start-btns"
             id="start-btns-hard"></div>
      </div>
    </div>
  </div>

  <hr class="divider">

  <button class="btn-run" id="btn-run"
          onclick="runOptimiser()">
    Optimise Strategy
  </button>

  <div class="status-box" id="status-box"></div>
</div>

<script>
let scSelected = 'low';
let tyreMode   = {
  soft: 'unlimited', medium: 'unlimited', hard: 'unlimited'
};
let startingSet = { soft: -1, medium: -1, hard: -1 };

async function loadCircuits() {
  const res  = await fetch('/circuits');
  const data = await res.json();
  const sel  = document.getElementById('circuit');
  data.circuits.forEach(function(c) {
    const opt  = document.createElement('option');
    opt.value  = c;
    opt.text   = c;
    if (c === 'Bahrain Grand Prix') opt.selected = true;
    sel.appendChild(opt);
  });
}

function selectSC(level) {
  scSelected = level;
  ['low','medium','high'].forEach(function(l) {
    const el = document.getElementById('sc-' + l);
    el.className = 'sc-btn' + (
      l === level ? ' selected-' + l : '');
  });
}

function setMode(compound, mode) {
  tyreMode[compound] = mode;
  const inputsEl = document.getElementById(
    'inputs-' + compound);
  const unlBtn   = document.getElementById(
    'toggle-' + compound + '-unlimited');
  const spBtn    = document.getElementById(
    'toggle-' + compound + '-specify');

  if (mode === 'unlimited') {
    inputsEl.classList.remove('visible');
    unlBtn.classList.add('active');
    spBtn.classList.remove('active');
    startingSet[compound] = -1;
  } else {
    inputsEl.classList.add('visible');
    spBtn.classList.add('active');
    unlBtn.classList.remove('active');
    updateAgeInputs(compound);
  }
}

function updateAgeInputs(compound) {
  const n         = parseInt(
    document.getElementById(
      'sets-' + compound).value) || 0;
  const container = document.getElementById(
    'ages-' + compound);
  const startRow  = document.getElementById(
    'start-row-' + compound);

  container.innerHTML = '';

  for (let i = 0; i < n; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'age-input-wrap';
    const lbl  = document.createElement('label');
    lbl.textContent = 'Set ' + (i + 1);
    const inp  = document.createElement('input');
    inp.type   = 'number';
    inp.id     = 'age-' + compound + '-' + i;
    inp.value  = '0';
    inp.min    = '0';
    inp.max    = '50';
    inp.placeholder = 'laps';
    inp.oninput = function() {
      updateStartBtns(compound);
    };
    wrap.appendChild(lbl);
    wrap.appendChild(inp);
    container.appendChild(wrap);
  }

  if (n > 0) {
    startRow.style.display = 'block';
    startingSet[compound]  = -1;
    updateStartBtns(compound);
  } else {
    startRow.style.display = 'none';
    startingSet[compound]  = -1;
  }
}

function updateStartBtns(compound) {
  const n         = parseInt(
    document.getElementById(
      'sets-' + compound).value) || 0;
  const container = document.getElementById(
    'start-btns-' + compound);
  const current   = startingSet[compound];

  container.innerHTML = '';

  for (let i = 0; i < n; i++) {
    const ageEl = document.getElementById(
      'age-' + compound + '-' + i);
    const age   = ageEl
                  ? (parseInt(ageEl.value) || 0) : 0;
    const btn   = document.createElement('div');
    btn.className = 'start-btn' + (
      i === current ? ' selected' : '');
    btn.textContent = 'Set ' + (i + 1) +
                      ' (' + age + ' laps)';
    btn.onclick = (function(idx) {
      return function() {
        selectStartingSet(compound, idx);
      };
    })(i);
    container.appendChild(btn);
  }

  if (current === -1 && n > 0) {
    selectStartingSet(compound, 0);
  }
}

function selectStartingSet(compound, idx) {
  ['soft', 'medium', 'hard'].forEach(function(c) {
    if (c !== compound && startingSet[c] !== -1) {
      startingSet[c] = -1;
      updateStartBtns(c);
    }
  });
  startingSet[compound] = idx;
  updateStartBtns(compound);
}

function getTyreData() {
  const result  = { sets: {}, start_age: 0 };
  const compMap = {
    soft: 'SOFT', medium: 'MEDIUM', hard: 'HARD'
  };

  ['soft', 'medium', 'hard'].forEach(function(c) {
    if (tyreMode[c] !== 'specify') return;

    const n = parseInt(
      document.getElementById('sets-' + c).value) || 0;
    if (n > 0) {
      result.sets[compMap[c]] = n;
    }

    const idx = startingSet[c];
    if (idx >= 0) {
      const ageEl = document.getElementById(
        'age-' + c + '-' + idx);
      if (ageEl) {
        result.start_age = parseInt(ageEl.value) || 0;
      }
    }
  });

  return result;
}

function setStatus(msg, type) {
  type = type || 'active';
  const box  = document.getElementById('status-box');
  box.style.display = 'block';
  const line = document.createElement('div');
  line.className   = 'status-line ' + type;
  line.textContent = '> ' + msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function clearStatus() {
  const box = document.getElementById('status-box');
  box.innerHTML     = '';
  box.style.display = 'none';
}

async function runOptimiser() {
  clearStatus();
  const circuit  = document.getElementById('circuit').value;
  const laps     = parseInt(
    document.getElementById('laps').value);
  const tyreData = getTyreData();

  if (!circuit) {
    setStatus('Select a circuit', 'error'); return;
  }
  if (laps < 20 || laps > 80) {
    setStatus('Laps must be between 20 and 80', 'error');
    return;
  }

  document.getElementById('btn-run').disabled = true;
  setStatus('Optimising ' + circuit +
            ' (' + laps + ' laps)...');

  try {
    const res  = await fetch('/optimise', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({
        circuit:   circuit,
        laps:      laps,
        sc:        scSelected,
        tyre_sets: Object.keys(tyreData.sets).length > 0
                   ? tyreData.sets : null,
        start_age: tyreData.start_age,
      })
    });
    const data = await res.json();

    if (data.error) {
      setStatus(data.error, 'error');
      document.getElementById('btn-run').disabled = false;
      return;
    }

    setStatus('Done — opening results', 'done');
    setTimeout(function() {
      window.open('/results', '_blank');
      document.getElementById('btn-run').disabled = false;
    }, 600);

  } catch(e) {
    setStatus('Error: ' + e.message, 'error');
    document.getElementById('btn-run').disabled = false;
  }
}

loadCircuits();
</script>
</body>
</html>'''

results_store = {'html': None}


def build_results_html(circuit, laps, sc,
                       strategies, model,
                       tyre_sets=None, start_age=0):
    C = COMPOUND_COLOURS

    def fmt_time(seconds):
        m = int(seconds // 60)
        s = seconds % 60
        return f'{m}m {s:.3f}s'

    pit_loss = model['pit_loss'].get(circuit, 22.0)
    deg_data = model['degradation'].get(circuit, {})
    avail    = [c for c in ['SOFT', 'MEDIUM', 'HARD']
                if c in deg_data]

    sc_label = {
        'low':    'LOW',
        'medium': 'MEDIUM',
        'high':   'HIGH',
    }.get(sc, sc.upper())

    COMPOUND_HEX = {
        'SOFT':   '#e8002d',
        'MEDIUM': '#ffd700',
        'HARD':   '#aaaaaa',
    }

    def build_lap_chart(strategies, total_laps):
        rows = ''
        for i, s in enumerate(strategies[:5]):
            is_best    = i == 0
            compounds  = s['compounds']
            stint_laps = s['stint_laps']
            delta      = s['delta_vs_optimal']

            label     = ' → '.join(compounds)
            delta_str = (
                'OPTIMAL' if delta == 0
                else f'+{delta:.3f}s')
            label_col = '#e8002d' if is_best else '#aaa'

            blocks = ''
            cur    = 0
            for k, (compound, stint_len) in enumerate(
                    zip(compounds, stint_laps)):
                pct      = (stint_len / total_laps) * 100
                col      = COMPOUND_HEX.get(compound, '#888')
                label_txt = (compound if stint_len > 8
                             else compound[:1])
                txt_col  = (
                    '#000' if compound in ('HARD', 'MEDIUM')
                    else '#fff')
                blocks += (
                    f'<div style="position:relative;'
                    f'width:{pct:.2f}%;height:100%;'
                    f'background:{col};display:inline-flex;'
                    f'align-items:center;'
                    f'justify-content:center;'
                    f'border-right:1px solid #0a0a0a;'
                    f'overflow:hidden;">'
                    f'<span style="font-size:9px;'
                    f'font-weight:700;color:{txt_col};">'
                    f'{label_txt}</span></div>'
                )
                cur += stint_len

            markers = ''
            cur = 0
            for k, stint_len in enumerate(stint_laps[:-1]):
                cur += stint_len
                pct  = (cur / total_laps) * 100
                markers += (
                    f'<div style="position:absolute;'
                    f'left:{pct:.2f}%;top:0;bottom:0;'
                    f'width:2px;background:#fff;'
                    f'z-index:2;"></div>'
                )

            rows += f'''
            <tr style="border-bottom:1px solid #1a1a1a;">
              <td style="padding:8px 12px;white-space:nowrap;
                         font-size:11px;color:{label_col};
                         font-weight:{'700' if is_best
                                      else '400'};
                         min-width:180px;">
                {'★ ' if is_best else ''}{label}<br>
                <span style="color:#aaa;font-size:10px;
                             font-weight:400;">
                  {delta_str}</span>
              </td>
              <td style="padding:4px 12px;width:100%;">
                <div style="position:relative;height:28px;
                            background:#1a1a1a;
                            border-radius:3px;overflow:hidden;
                            display:flex;">
                  {blocks}{markers}
                </div>
              </td>
              <td style="padding:8px 12px;white-space:nowrap;
                         font-size:11px;color:#ccc;
                         min-width:120px;">
                {fmt_time(s['total_time'])}
              </td>
            </tr>'''

        window_info = ''
        for i, s in enumerate(strategies[:5]):
            windows   = s.get('pit_windows', [])
            if not windows:
                continue
            compounds = s['compounds']
            label     = ' → '.join(compounds)
            parts     = []
            for k, w in enumerate(windows):
                if w:
                    parts.append(
                        f'Stop {k+1}: Lap {w[0]}–{w[1]}')
            if parts:
                is_best   = i == 0
                label_col = '#e8002d' if is_best else '#aaa'
                window_info += (
                    f'<div style="font-size:11px;'
                    f'margin-bottom:6px;">'
                    f'<span style="color:{label_col};">'
                    f'{label}</span>'
                    f'<span style="color:#ccc;'
                    f'margin-left:12px;">'
                    + ' · '.join(parts)
                    + f'</span></div>'
                )

        legend = (
            '<div style="display:flex;gap:16px;'
            'margin-bottom:12px;align-items:center;">'
        )
        for compound, col in COMPOUND_HEX.items():
            txt = ('#000' if compound in ('HARD', 'MEDIUM')
                   else '#fff')
            if compound in avail:
                legend += (
                    f'<span style="background:{col};'
                    f'color:{txt};padding:2px 8px;'
                    f'border-radius:3px;font-size:10px;'
                    f'font-weight:700;">{compound}</span>'
                )
        legend += (
            '<span style="font-size:10px;color:#aaa;">'
            '│ <span style="display:inline-block;'
            'width:12px;height:12px;background:#fff;'
            'vertical-align:middle;margin-right:4px;'
            'opacity:0.8;"></span>Pit stop'
            '</span></div>'
        )

        return f'''
        <div style="margin-bottom:32px;">
          {legend}
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr style="border-bottom:1px solid #2a2a2a;">
                <th style="padding:6px 12px;text-align:left;
                    font-size:10px;color:#aaa;font-weight:600;
                    text-transform:uppercase;
                    letter-spacing:1px;">Strategy</th>
                <th style="padding:6px 12px;text-align:left;
                    font-size:10px;color:#aaa;font-weight:600;
                    text-transform:uppercase;
                    letter-spacing:1px;">Lap Chart</th>
                <th style="padding:6px 12px;text-align:left;
                    font-size:10px;color:#aaa;font-weight:600;
                    text-transform:uppercase;
                    letter-spacing:1px;">Race Time</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
          <div style="margin-top:16px;padding:12px;
               background:#0f0f0f;border:1px solid #1e1e1e;
               border-radius:6px;">
            <div style="font-size:10px;color:#aaa;
                 text-transform:uppercase;letter-spacing:1px;
                 margin-bottom:8px;">
              Pit Windows (±1.5s)</div>
            {window_info}
          </div>
        </div>'''

    lap_chart_html = build_lap_chart(strategies, laps)

    # Tyre availability summary
    avail_str = ''
    if tyre_sets:
        parts = []
        for c, n in tyre_sets.items():
            col  = COMPOUND_HEX.get(c, '#888')
            tcol = ('#000' if c in ('HARD', 'MEDIUM')
                    else '#fff')
            parts.append(
                f'<span style="background:{col};color:{tcol};'
                f'padding:2px 8px;border-radius:3px;'
                f'font-size:10px;font-weight:700;">'
                f'{c} ×{n}</span>')
        if start_age > 0:
            parts.append(
                f'<span style="color:#aaa;font-size:10px;">'
                f'Starting age: {start_age} laps</span>')
        avail_str = (
            '<div style="display:flex;gap:8px;'
            'flex-wrap:wrap;margin-bottom:32px;">'
            + ' '.join(parts) + '</div>')

    # Degradation table
    deg_rows = ''
    for c in avail:
        d     = deg_data[c]
        col   = C.get(c, '#888')
        tcol  = '#000' if c in ('HARD', 'MEDIUM') else '#fff'
        src   = d.get('pace_source', '?')
        ms    = model.get('max_stints', {}).get(
            circuit, {}).get(c, '?')
        cliff = model.get('cliff_model', {}).get(
            circuit, {}).get(c, {})
        cliff_str = ''
        if cliff.get('has_cliff'):
            tf     = cliff.get('temp_factor', '')
            tf_str = f' temp×{tf}' if tf else ''
            cliff_str = (
                f' cliff@{cliff["cliff_lap"]} '
                f'(×{cliff["cliff_ratio"]:.1f}{tf_str})')
        deg_rows += (
            f'<tr>'
            f'<td style="padding:6px 12px;">'
            f'<span style="background:{col};color:{tcol};'
            f'padding:2px 8px;border-radius:3px;'
            f'font-size:11px;font-weight:700;">'
            f'{c}</span></td>'
            f'<td style="padding:6px 12px;color:#ddd;">'
            f'{d["base_pace"]:.3f}s</td>'
            f'<td style="padding:6px 12px;color:#ddd;">'
            f'+{d["deg_rate"]:.4f}s/lap</td>'
            f'<td style="padding:6px 12px;color:#ccc;">'
            f'{d.get("fuel_rate",0):+.4f}s/lap</td>'
            f'<td style="padding:6px 12px;color:#ccc;">'
            f'R²={d["r_squared"]:.3f}</td>'
            f'<td style="padding:6px 12px;color:#ccc;">'
            f'n={d["n_points"]}</td>'
            f'<td style="padding:6px 12px;color:#ccc;">'
            f'max={ms}laps{cliff_str}</td>'
            f'<td style="padding:6px 12px;color:#888;'
            f'font-size:10px;">[{src}]</td>'
            f'</tr>'
        )

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Strategy — {circuit}</title>
<style>
* {{ box-sizing:border-box;margin:0;padding:0; }}
html,body {{ background:#0a0a0a;color:#fff;
  font-family:Segoe UI,sans-serif;min-height:100%; }}
#header {{ padding:16px 32px;
  border-bottom:1px solid #1a1a1a;
  display:flex;align-items:center;
  justify-content:space-between; }}
#header h1 {{ font-size:15px;font-weight:500; }}
#header p  {{ font-size:11px;color:#aaa;margin-top:3px; }}
.mode {{ font-size:10px;color:#888;letter-spacing:1px;
  text-transform:uppercase; }}
.body {{ max-width:1000px;margin:0 auto;padding:32px 24px; }}
.section-title {{ font-size:11px;font-weight:600;
  letter-spacing:2px;text-transform:uppercase;
  color:#aaa;margin-bottom:16px; }}
.info-row {{ display:flex;gap:32px;
  margin-bottom:32px;flex-wrap:wrap; }}
.info-item {{ display:flex;flex-direction:column;gap:4px; }}
.info-label {{ font-size:10px;color:#aaa;
  text-transform:uppercase;letter-spacing:1px; }}
.info-value {{ font-size:14px;font-weight:600; }}
.disclaimer {{
  background:#0f0f0f;border:1px solid #1e1e1e;
  border-radius:6px;padding:16px;margin-bottom:32px;
  font-size:11px;color:#bbb;line-height:1.7;
}}
</style>
</head>
<body>
<div id="header">
  <div>
    <h1>{circuit} — {laps} laps</h1>
    <p>Strategy optimiser · Safety car: {sc_label} ·
       Pit loss: {pit_loss}s</p>
  </div>
  <span class="mode">F1 STRATEGY OPTIMISER</span>
</div>

<div class="body">
  <div class="info-row">
    <div class="info-item">
      <span class="info-label">Circuit</span>
      <span class="info-value">{circuit}</span>
    </div>
    <div class="info-item">
      <span class="info-label">Race Distance</span>
      <span class="info-value">{laps} laps</span>
    </div>
    <div class="info-item">
      <span class="info-label">Safety Car</span>
      <span class="info-value">{sc_label}</span>
    </div>
    <div class="info-item">
      <span class="info-label">Pit Loss</span>
      <span class="info-value">{pit_loss}s</span>
    </div>
    <div class="info-item">
      <span class="info-label">Compounds</span>
      <span class="info-value">{' / '.join(avail)}</span>
    </div>
  </div>

  {avail_str}

  <div class="disclaimer">
    <strong style="color:#ddd;">Model notes</strong> —
    Degradation fitted from 112,732 stint-laps across
    2020–2025 using quadratic multiple regression
    (separating fuel burn from tyre wear).
    Piecewise cliff model fitted per compound per circuit.
    Base pace corrected using FP2 session data,
    validated against race pace gaps.
    Pirelli circuit ratings applied as fallback for
    data-sparse circuits. Track temperature scaling
    applied to cliff degradation rates.
    Pit windows show lap range within 1.5s of optimal.
    Known limitation: at high-lateral circuits (Suzuka,
    Zandvoort) the model may underweight MEDIUM→HARD
    strategies — tyre temperature management behaviour
    is not captured in public FastF1 timing data.
    Safety car probability scales pit loss by
    1.0× / 0.85× / 0.70×.
  </div>

  <div class="section-title">
    Top 5 Strategies — Lap Chart</div>
  {lap_chart_html}

  <div style="margin-top:32px;">
    <div class="section-title">
      Tyre Model — {circuit}</div>
    <div style="background:#0f0f0f;
         border:1px solid #1e1e1e;
         border-radius:8px;overflow:hidden;">
      <table style="width:100%;border-collapse:collapse;
             font-size:12px;">
        <thead>
          <tr style="border-bottom:1px solid #1a1a1a;">
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">Compound</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">Base Pace</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">Deg Rate</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">Fuel Rate</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">R²</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">
                Data Points</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">
                Max Stint</th>
            <th style="padding:10px 12px;text-align:left;
                color:#aaa;font-weight:600;">Source</th>
          </tr>
        </thead>
        <tbody>{deg_rows}</tbody>
      </table>
    </div>
  </div>
</div>
</body>
</html>'''

    return html


@app.route('/')
def index():
    return LAUNCHER_HTML


@app.route('/circuits')
def circuits():
    try:
        model    = load_model()
        circuits = sorted(model['degradation'].keys())
        return jsonify({'circuits': circuits})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/optimise', methods=['POST'])
def run_optimise():
    body      = request.get_json()
    circuit   = body.get('circuit')
    laps      = int(body.get('laps', 57))
    sc        = body.get('sc', 'low')
    tyre_sets = body.get('tyre_sets', None)
    start_age = int(body.get('start_age', 0))

    try:
        model      = load_model()
        strategies = optimise(
            circuit, laps, sc,
            top_n=15,
            tyre_sets=tyre_sets,
            start_age=start_age)

        if not strategies:
            return jsonify({
                'error': (
                    f'No strategies found for {circuit}. '
                    f'Check tyre availability settings.')
            })

        html = build_results_html(
            circuit, laps, sc, strategies, model,
            tyre_sets=tyre_sets, start_age=start_age)
        results_store['html'] = html

        return jsonify({'ok': True})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/results')
def results():
    if results_store['html']:
        return results_store['html']
    return 'No results yet', 404


if __name__ == '__main__':
    import threading
    import webbrowser
    import time

    port     = int(os.environ.get('PORT', 5001))
    is_local = port == 5001

    if is_local:
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://127.0.0.1:{port}')
        threading.Thread(
            target=open_browser, daemon=True).start()
        print(f'F1 Strategy Optimiser starting...')
        print(f'http://127.0.0.1:{port}')

    app.run(debug=False, host='0.0.0.0', port=port)