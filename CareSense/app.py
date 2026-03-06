
import os, ast, pickle, io, json, datetime, re
import pandas as pd
import numpy as np
from flask import (Flask, render_template, request, jsonify, send_file, session)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image as RLImage,
                                 KeepTogether, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus.flowables import Flowable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'caresense_secret_v2_2024'

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
GRAPH_DIR = os.path.join(BASE_DIR, 'graphs')

# ── Disease Name Alias Map ─────────────────────────────────────────────────────
# FIX #1 & #2: Map training disease names → lookup file disease names
DISEASE_ALIAS = {
    'Peptic ulcer diseae':                     'Peptic ulcer disease',
    '(vertigo) Paroymsal  Positional Vertigo': '(vertigo) Paroymsal Positional Vertigo',
}

def normalize_disease(name: str) -> str:
    """Resolve typos/double-spaces in disease names from training data."""
    stripped = name.strip()
    # Try exact alias
    if stripped in DISEASE_ALIAS:
        return DISEASE_ALIAS[stripped]
    # Try collapsing multiple spaces
    collapsed = re.sub(r'\s+', ' ', stripped)
    if collapsed in DISEASE_ALIAS.values():
        return collapsed
    return stripped

# ── Load ML Artifacts ──────────────────────────────────────────────────────────
def _pkl(name):
    with open(os.path.join(MODEL_DIR, name), 'rb') as f:
        return pickle.load(f)

def _json(name):
    with open(os.path.join(MODEL_DIR, name)) as f:
        return json.load(f)

model            = _pkl('best_model.pkl')
le               = _pkl('label_encoder.pkl')
symptoms_list    = _pkl('symptoms_list.pkl')
sev_weights      = _pkl('severity_weights.pkl')
ensemble_bundle  = _pkl('ensemble.pkl')           # RF+SVM+NB soft-voting
cooc_map         = _json('cooccurrence.json')
key_sym_map      = _json('disease_key_symptoms.json')
disease_sym_freq = _json('disease_sym_freq.json') # per-disease symptom frequencies
emergency_combos = _json('emergency_combos.json')
threshold_cfg    = _json('threshold_config.json')
body_regions     = _json('body_regions.json')     # body map region → symptoms

# ── Risk Factor Probability Boosts ─────────────────────────────────────────────
# Maps risk_factor_key → {disease_name: multiplier}
RISK_BOOSTS = {
    'diabetic':     {'Diabetes': 1.6, 'Hypertension': 1.2, 'Heart attack': 1.2,
                     'Hypoglycemia': 1.3, 'Hyperthyroidism': 1.1},
    'smoker':       {'Bronchial Asthma': 1.5, 'Pneumonia': 1.3, 'Tuberculosis': 1.3,
                     'Chronic cholestasis': 1.1, 'GERD': 1.1},
    'hypertensive': {'Hypertension': 1.5, 'Heart attack': 1.3, 'Paralysis (brain hemorrhage)': 1.2},
    'age_under18':  {'Chicken pox': 1.4, 'Malaria': 1.2, 'Typhoid': 1.2, 'Dengue': 1.2},
    'age_51_70':    {'Arthritis': 1.3, 'Heart attack': 1.2, 'Diabetes': 1.1,
                     'Osteoarthristis': 1.3, 'Hypertension': 1.2},
    'age_70plus':   {'Arthritis': 1.5, 'Heart attack': 1.4, 'Diabetes': 1.2,
                     'Osteoarthristis': 1.4, 'Hypertension': 1.3, 'Paralysis (brain hemorrhage)': 1.2},
}

# Duration multipliers: maps user-selected duration label → weight multiplier
DURATION_MULT = {
    'hours':   0.6,   # acute onset — may be less significant
    'days':    1.0,   # standard
    'weeks':   1.3,   # persisting — more clinically significant
    'chronic': 1.6,   # long-standing — highest clinical weight
}

# ── Load Data CSVs ─────────────────────────────────────────────────────────────
def _csv(name, **kw):
    return pd.read_csv(os.path.join(DATA_DIR, name), **kw)

desc_df  = _csv('description.csv');   desc_df.columns  = desc_df.columns.str.strip()
med_df   = _csv('medications.csv');   med_df.columns   = med_df.columns.str.strip()
prec_df  = _csv('precautions_df.csv'); prec_df.columns = prec_df.columns.str.strip()
diet_df  = _csv('diets.csv');          diet_df.columns  = diet_df.columns.str.strip()
wout_df  = _csv('workout_df.csv');     wout_df.columns  = wout_df.columns.str.strip()

met_df = pd.read_csv(os.path.join(MODEL_DIR, 'model_metrics.csv')) \
         if os.path.exists(os.path.join(MODEL_DIR, 'model_metrics.csv')) else pd.DataFrame()

def _build_dict(df, key_col, val_col):
    return {str(r[key_col]).strip(): r[val_col] for _, r in df.iterrows()}

desc_dict = _build_dict(desc_df, 'Disease', 'Description')
med_dict  = _build_dict(med_df,  'Disease', 'Medication')
diet_dict = _build_dict(diet_df, 'Disease', 'Diet')

def _build_prec(df):
    out = {}
    for _, r in df.iterrows():
        k = str(r.get('Disease', '')).strip()
        vals = [str(r.get(f'Precaution_{i}', '')).strip()
                for i in range(1, 5)
                if str(r.get(f'Precaution_{i}', '')).strip() not in ('', 'nan')]
        out[k] = vals
    return out
prec_dict = _build_prec(prec_df)

def _build_wout(df):
    out = {}
    col = 'disease' if 'disease' in df.columns else df.columns[1]
    for disease, grp in df.groupby(col):
        out[disease.strip()] = grp['workout'].dropna().tolist()
    return out
wout_dict = _build_wout(wout_df)

# Sorted symptom list for UI (exclude any accidental 'prognosis' entry)
all_symptoms = sorted([s.strip() for s in symptoms_list if s.strip() != 'prognosis'])

# Normalise severity weights: max weight for scaling
max_sev_w = max((v for k, v in sev_weights.items() if k != 'prognosis'), default=1)

# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_symptom(sym: str) -> str:
    """Normalize symptom string for weight lookup."""
    return sym.strip()

def get_symptom_weight(sym: str) -> float:
    """FIX #3: Look up weight with fallback to 1."""
    s = normalize_symptom(sym)
    return sev_weights.get(s, 1)

def get_dynamic_threshold(severity_score: int):
    """FIX #7: Return (threshold, label) based on symptom severity score."""
    for step in threshold_cfg['severity_steps']:
        if step['min_score'] <= severity_score <= step['max_score']:
            return step['threshold'], step['label']
    return threshold_cfg['base_threshold'], 'Unknown'

def compute_severity(selected_symptoms: list, duration_weights: dict = None) -> int:
    """Sum severity weights, optionally scaled by duration multipliers."""
    total = 0
    for s in selected_symptoms:
        base = get_symptom_weight(s)
        mult = 1.0
        if duration_weights:
            dur_label = duration_weights.get(s, 'days')
            mult = DURATION_MULT.get(dur_label, 1.0)
        total += base * mult
    return round(total)

def predict_disease(selected_symptoms: list, duration_weights: dict = None):
    """Ensemble soft-voting: RF(0.50) + SVM(0.35) + NB(0.15).
    duration_weights: {symptom: 'hours'|'days'|'weeks'|'chronic'} — scales feature values.
    """
    input_vec = np.zeros(len(symptoms_list))
    for sym in selected_symptoms:
        s = sym.strip()
        if s in symptoms_list:
            base_w = get_symptom_weight(s) / max_sev_w
            dur_label = (duration_weights or {}).get(s, 'days')
            dur_mult  = DURATION_MULT.get(dur_label, 1.0)
            input_vec[symptoms_list.index(s)] = base_w * dur_mult

    ens_models  = ensemble_bundle['models']
    ens_weights = ensemble_bundle['weights']

    # Weighted average of per-model probability vectors
    avg_proba = np.zeros(len(le.classes_))
    model_votes = {}
    for name, mdl in ens_models.items():
        proba = mdl.predict_proba([input_vec])[0]
        avg_proba += ens_weights[name] * proba
        top_i = int(np.argmax(proba))
        model_votes[name] = {
            'disease':    le.classes_[top_i],
            'confidence': round(float(proba[top_i]) * 100, 1),
        }

    top_idx    = np.argsort(avg_proba)[::-1]
    disease    = le.classes_[top_idx[0]]
    confidence = float(avg_proba[top_idx[0]])
    top5       = {le.classes_[i]: round(float(avg_proba[i]) * 100, 2)
                  for i in top_idx[:5]}

    return disease, confidence, top5, model_votes


def apply_risk_boosts(top5: dict, risk_factors: dict) -> dict:
    """Post-process top5 probabilities based on user risk factors.
    Boosts relevant disease probabilities then re-normalises the top5.
    """
    if not risk_factors:
        return top5

    boosted = dict(top5)
    boost_keys = []
    if risk_factors.get('diabetic'):     boost_keys.append('diabetic')
    if risk_factors.get('smoker'):       boost_keys.append('smoker')
    if risk_factors.get('hypertensive'): boost_keys.append('hypertensive')
    age = risk_factors.get('age_group', '18-30')
    if age == 'under-18': boost_keys.append('age_under18')
    if age == '51-70':   boost_keys.append('age_51_70')
    if age == '70+':     boost_keys.append('age_70plus')

    if not boost_keys:
        return top5

    for key in boost_keys:
        boosts = RISK_BOOSTS.get(key, {})
        for disease, mult in boosts.items():
            if disease in boosted:
                boosted[disease] = round(boosted[disease] * mult, 2)

    # Cap at 99.9 and re-sort
    boosted = {k: min(v, 99.9) for k, v in boosted.items()}
    boosted = dict(sorted(boosted.items(), key=lambda x: -x[1]))
    return boosted


def get_differential_explanation(user_symptoms: list, top5: dict) -> dict:
    """
    For each top-5 disease return:
      matching   — user symptoms that appear in this disease (with strength label)
      missing    — key disease symptoms the user did NOT report (up to 3)
      match_pct  — fraction of user symptoms that point to this disease
    """
    user_set  = {s.strip() for s in user_symptoms}
    explained = {}

    for disease in top5:
        d_key   = disease.strip()
        freq_map = disease_sym_freq.get(d_key, {})
        key_list = key_sym_map.get(d_key, [])

        # Classify each user symptom by how strongly it points to this disease
        strong   = []   # ≥ 70 % of training rows for this disease
        moderate = []   # 30–69 %
        weak     = []   # > 0 – 29 %

        for sym in user_symptoms:
            s = sym.strip()
            f = freq_map.get(s, 0)
            if   f >= 0.70: strong.append(s)
            elif f >= 0.30: moderate.append(s)
            elif f >  0.00: weak.append(s)

        # Key symptoms the user didn't mention (top 3, ordered by freq desc)
        missing = sorted(
            [s for s in key_list if s not in user_set and s != 'prognosis'],
            key=lambda s: -freq_map.get(s, 0)
        )[:3]

        total   = len(user_symptoms)
        matched = len(strong) + len(moderate) + len(weak)
        match_pct = round(matched / total * 100) if total else 0

        explained[disease] = {
            'strong':    strong,
            'moderate':  moderate,
            'weak':      weak,
            'missing':   missing,
            'match_pct': match_pct,
        }

    return explained

def get_disease_info(disease: str) -> dict:
    """FIX #11: Use alias map so all 41 diseases return complete info."""
    raw = disease.strip()
    d   = normalize_disease(raw)   # resolve Peptic/Vertigo typos

    def _parse_list(raw_val):
        try:
            result = ast.literal_eval(raw_val)
            return result if isinstance(result, list) else [str(result)]
        except Exception:
            return [str(raw_val)] if raw_val and raw_val not in ('[]', 'nan') else []

    meds  = _parse_list(med_dict.get(d, med_dict.get(raw, '[]')))
    diets = _parse_list(diet_dict.get(d, diet_dict.get(raw, '[]')))

    return {
        'disease':     raw,
        'description': desc_dict.get(d, desc_dict.get(raw, 'No description available.')),
        'medications': meds,
        'precautions': prec_dict.get(d, prec_dict.get(raw, [])),
        'diets':       diets,
        'workouts':    wout_dict.get(d, wout_dict.get(raw, [])),
    }

def check_emergency(selected_symptoms: list):
    """#3: Return first matching emergency combo, or None."""
    sym_set = {normalize_symptom(s).lower() for s in selected_symptoms}
    for combo in emergency_combos:
        if set(combo['symptoms']).issubset(sym_set):
            return combo
    return None

def get_suggestions(selected_symptoms: list, limit: int = 6) -> list:
    """#5: Return co-occurring symptoms not yet selected."""
    selected_set = {normalize_symptom(s) for s in selected_symptoms}
    score: dict = {}
    for sym in selected_symptoms:
        for related in cooc_map.get(normalize_symptom(sym), []):
            if related and related != 'prognosis' and related not in selected_set:
                score[related] = score.get(related, 0) + 1
    return sorted(score, key=lambda k: -score[k])[:limit]

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', symptoms=all_symptoms, body_regions=body_regions)

@app.route('/predict', methods=['POST'])
def predict():
    data             = request.get_json(silent=True) or {}
    selected         = [s for s in data.get('symptoms', []) if s.strip()]
    duration_weights = data.get('duration_weights', {})   # {symptom: 'hours'|'days'|'weeks'|'chronic'}
    risk_factors     = data.get('risk_factors', {})        # {diabetic, smoker, hypertensive, age_group}

    if not selected:
        return jsonify({'error': 'Please select at least 1 symptom.'}), 400

    emergency = check_emergency(selected)
    if emergency:
        disease, confidence, top5, model_votes = predict_disease(selected, duration_weights)
        top5 = apply_risk_boosts(top5, risk_factors)
        disease = next(iter(top5))  # top disease after boost
        sev_score = compute_severity(selected, duration_weights)
        threshold, sev_label = get_dynamic_threshold(sev_score)
        info = get_disease_info(disease)
        suggestions  = get_suggestions(selected)
        differential = get_differential_explanation(selected, top5)
        selected_set = {normalize_symptom(s) for s in selected}
        followup_qs  = [s for s in key_sym_map.get(disease.strip(), [])
                        if s not in selected_set and s != 'prognosis'][:3]
        return jsonify({
            'emergency':         True,
            'emergency_info':    emergency,
            'disease':           disease,
            'confidence':        round(confidence * 100, 2),
            'top5':              top5,
            'differential':      differential,
            'model_votes':       model_votes,
            'consult_doctor':    True,
            'threshold_used':    round(threshold * 100, 1),
            'severity_score':    sev_score,
            'severity_label':    sev_label,
            'sym_count':         len(selected),
            'info':              info,
            'selected_symptoms': selected,
            'duration_weights':  duration_weights,
            'risk_factors':      risk_factors,
            'suggestions':       suggestions,
            'followup_qs':       followup_qs,
        })

    disease, confidence, top5, model_votes = predict_disease(selected, duration_weights)
    top5    = apply_risk_boosts(top5, risk_factors)
    disease = next(iter(top5))
    sev_score  = compute_severity(selected, duration_weights)
    threshold, sev_label = get_dynamic_threshold(sev_score)
    consult = confidence < threshold
    info    = get_disease_info(disease)
    suggestions  = get_suggestions(selected)
    differential = get_differential_explanation(selected, top5)
    selected_set = {normalize_symptom(s) for s in selected}
    followup_qs  = [s for s in key_sym_map.get(disease.strip(), [])
                    if s not in selected_set and s != 'prognosis'][:3]

    result = {
        'emergency':         False,
        'disease':           disease,
        'confidence':        round(confidence * 100, 2),
        'top5':              top5,
        'differential':      differential,
        'model_votes':       model_votes,
        'consult_doctor':    consult,
        'threshold_used':    round(threshold * 100, 1),
        'severity_score':    sev_score,
        'severity_label':    sev_label,
        'sym_count':         len(selected),
        'info':              info,
        'selected_symptoms': selected,
        'duration_weights':  duration_weights,
        'risk_factors':      risk_factors,
        'suggestions':       suggestions,
        'followup_qs':       followup_qs,
    }
    session['last_result'] = result
    return jsonify(result)

@app.route('/followup', methods=['POST'])
def followup():
    """Re-predict after follow-up confirmation."""
    data             = request.get_json(silent=True) or {}
    original         = data.get('original_symptoms', [])
    confirmed        = data.get('confirmed_symptoms', [])
    duration_weights = data.get('duration_weights', {})
    risk_factors     = data.get('risk_factors', {})
    all_syms         = list({normalize_symptom(s) for s in original + confirmed})

    emergency = check_emergency(all_syms)
    if emergency:
        disease, confidence, top5, model_votes = predict_disease(all_syms, duration_weights)
        top5 = apply_risk_boosts(top5, risk_factors)
        disease = next(iter(top5))
        sev_score = compute_severity(all_syms, duration_weights)
        threshold, sev_label = get_dynamic_threshold(sev_score)
        info = get_disease_info(disease)
        differential = get_differential_explanation(all_syms, top5)
        return jsonify({
            'emergency': True, 'emergency_info': emergency,
            'disease': disease, 'confidence': round(confidence*100, 2),
            'top5': top5, 'differential': differential, 'model_votes': model_votes,
            'consult_doctor': True,
            'threshold_used': round(threshold*100, 1),
            'severity_score': sev_score, 'severity_label': sev_label,
            'sym_count': len(all_syms), 'info': info,
            'selected_symptoms': all_syms, 'suggestions': [],
            'duration_weights': duration_weights, 'risk_factors': risk_factors,
            'followup_qs': [], 'validated': True,
        })

    disease, confidence, top5, model_votes = predict_disease(all_syms, duration_weights)
    top5    = apply_risk_boosts(top5, risk_factors)
    disease = next(iter(top5))
    sev_score = compute_severity(all_syms, duration_weights)
    threshold, sev_label = get_dynamic_threshold(sev_score)
    consult = confidence < threshold
    info    = get_disease_info(disease)
    suggestions  = get_suggestions(all_syms)
    differential = get_differential_explanation(all_syms, top5)

    result = {
        'emergency':         False,
        'disease':           disease,
        'confidence':        round(confidence * 100, 2),
        'top5':              top5,
        'differential':      differential,
        'model_votes':       model_votes,
        'consult_doctor':    consult,
        'threshold_used':    round(threshold * 100, 1),
        'severity_score':    sev_score,
        'severity_label':    sev_label,
        'sym_count':         len(all_syms),
        'info':              info,
        'selected_symptoms': all_syms,
        'suggestions':       suggestions,
        'duration_weights':  duration_weights,
        'risk_factors':      risk_factors,
        'followup_qs':       [],
        'validated':         True,
    }
    session['last_result'] = result
    return jsonify(result)

@app.route('/autocomplete')
def autocomplete():
    q = request.args.get('q', '').lower().strip()
    q_norm = q.replace(' ', '_')
    matches = [s for s in all_symptoms
               if q_norm in s.lower() or q in s.lower().replace('_', ' ')][:15]
    return jsonify(matches)

@app.route('/body_regions')
def get_body_regions():
    """Return body region → symptom mapping for the body map UI."""
    return jsonify(body_regions)

@app.route('/rediagnose', methods=['POST'])
def rediagnose():
    """Re-diagnose from result page: merge previous + new symptoms via /predict logic.
    Kept as a named route for clarity; delegates to the same predict pipeline.
    """
    data             = request.get_json(silent=True) or {}
    prev_syms        = data.get('previous_symptoms', [])
    new_syms         = data.get('new_symptoms', [])
    # Also accept flat 'symptoms' list for compatibility with JS calling /predict directly
    flat_syms        = data.get('symptoms', [])
    all_syms_raw     = flat_syms if flat_syms else prev_syms + new_syms
    duration_weights = data.get('duration_weights', {})
    risk_factors     = data.get('risk_factors', {})
    all_syms         = list({normalize_symptom(s) for s in all_syms_raw if s.strip()})

    if not all_syms:
        return jsonify({'error': 'No symptoms provided.'}), 400

    emergency = check_emergency(all_syms)
    if emergency:
        disease, confidence, top5, model_votes = predict_disease(all_syms, duration_weights)
        top5 = apply_risk_boosts(top5, risk_factors)
        disease = next(iter(top5))
        sev_score = compute_severity(all_syms, duration_weights)
        threshold, sev_label = get_dynamic_threshold(sev_score)
        info = get_disease_info(disease)
        differential = get_differential_explanation(all_syms, top5)
        selected_set = {normalize_symptom(s) for s in all_syms}
        followup_qs  = [s for s in key_sym_map.get(disease.strip(), [])
                        if s not in selected_set and s != 'prognosis'][:3]
        return jsonify({
            'emergency': True, 'emergency_info': emergency,
            'disease': disease, 'confidence': round(confidence*100, 2),
            'top5': top5, 'differential': differential, 'model_votes': model_votes,
            'consult_doctor': True, 'threshold_used': round(threshold*100, 1),
            'severity_score': sev_score, 'severity_label': sev_label,
            'sym_count': len(all_syms), 'info': info,
            'selected_symptoms': all_syms, 'duration_weights': duration_weights,
            'risk_factors': risk_factors, 'suggestions': get_suggestions(all_syms),
            'followup_qs': followup_qs,
        })

    disease, confidence, top5, model_votes = predict_disease(all_syms, duration_weights)
    top5    = apply_risk_boosts(top5, risk_factors)
    disease = next(iter(top5))
    sev_score = compute_severity(all_syms, duration_weights)
    threshold, sev_label = get_dynamic_threshold(sev_score)
    consult  = confidence < threshold
    info     = get_disease_info(disease)
    suggestions  = get_suggestions(all_syms)
    differential = get_differential_explanation(all_syms, top5)
    selected_set = {normalize_symptom(s) for s in all_syms}
    followup_qs  = [s for s in key_sym_map.get(disease.strip(), [])
                    if s not in selected_set and s != 'prognosis'][:3]

    result = {
        'emergency': False, 'disease': disease,
        'confidence': round(confidence * 100, 2),
        'top5': top5, 'differential': differential, 'model_votes': model_votes,
        'consult_doctor': consult, 'threshold_used': round(threshold * 100, 1),
        'severity_score': sev_score, 'severity_label': sev_label,
        'sym_count': len(all_syms), 'info': info,
        'selected_symptoms': all_syms, 'duration_weights': duration_weights,
        'risk_factors': risk_factors, 'suggestions': suggestions,
        'followup_qs': followup_qs,
    }
    session['last_result'] = result
    return jsonify(result)



@app.route('/suggestions', methods=['POST'])
def suggest():
    selected = request.get_json(silent=True) or {}
    selected = selected.get('symptoms', [])
    return jsonify(get_suggestions(selected))

@app.route('/analytics')
def analytics():
    graphs  = sorted([f for f in os.listdir(GRAPH_DIR) if f.endswith('.png')])
    metrics = met_df.to_dict('records') if not met_df.empty else []
    return render_template('analytics.html', graphs=graphs, metrics=metrics)

@app.route('/result', methods=['POST'])
def result_page():
    """Dedicated result page — receives JSON result from the diagnosis flow."""
    import json as _json
    result_json = request.form.get('result_json', '{}')
    try:
        result = _json.loads(result_json)
    except Exception:
        return redirect(url_for('index'))
    if not result or not result.get('disease'):
        return redirect(url_for('index'))
    session['last_result'] = result
    return render_template('result.html', r=result, all_symptoms=all_symptoms)

@app.route('/graphs/<filename>')
def serve_graph(filename):
    safe = os.path.basename(filename)
    path = os.path.join(GRAPH_DIR, safe)
    if not os.path.exists(path):
        return 'Not found', 404
    return send_file(path, mimetype='image/png')

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/diseases')
def diseases_page():
    """Diseases & Symptoms reference page."""
    # Reverse alias map: canonical name → raw training key
    rev_alias = {v: k for k, v in DISEASE_ALIAS.items()}
    disease_data = []
    for disease in sorted(desc_dict.keys()):
        info = get_disease_info(disease)
        # Try canonical name first, then the raw alias key
        raw_key  = rev_alias.get(disease, disease)
        key_syms = key_sym_map.get(disease, key_sym_map.get(raw_key, []))
        freq_map = disease_sym_freq.get(disease, disease_sym_freq.get(raw_key, {}))
        top_syms = sorted(freq_map.items(), key=lambda x: -x[1])[:8]
        disease_data.append({
            'name':         disease,
            'description':  info['description'],
            'medications':  info['medications'],
            'precautions':  info['precautions'],
            'diets':        info['diets'],
            'workouts':     info['workouts'],
            'key_symptoms': key_syms[:6],
            'top_symptoms': top_syms,
        })
    import pandas as _pd
    sev_df = _pd.read_csv(os.path.join(DATA_DIR, 'Symptom-severity.csv'))
    sev_df.columns = sev_df.columns.str.strip()
    sym_severity = sev_df.sort_values('weight', ascending=False).to_dict('records')
    return render_template('diseases.html',
                           diseases=disease_data,
                           sym_severity=sym_severity,
                           total_diseases=len(disease_data),
                           total_symptoms=len(sym_severity))

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    """FIX #7: Always read from POST body; session as secondary fallback."""
    result = request.get_json(silent=True)
    if not result:
        result = session.get('last_result')
    if not result:
        return jsonify({'error': 'No result data found. Please run a diagnosis first.'}), 400
    buf = io.BytesIO()
    generate_pdf(result, buf)
    buf.seek(0)
    disease_name = result.get('disease', 'report').strip().replace(' ', '_')
    fname = f"CareSense_{disease_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype='application/pdf')

# ── PDF Generation ─────────────────────────────────────────────────────────────

_C = {
    'green':     '#0a5c48',  'green_dk':  '#064e3b',
    'green_mid': '#6ee7b7',  'green_lt':  '#d1fae5',
    'green_xlt': '#ecfdf5',  'amber':     '#b45309',
    'amber_lt':  '#fef3c7',  'coral':     '#c0392b',
    'coral_lt':  '#fee2e2',  'violet':    '#5b3f9e',
    'violet_lt': '#ede9fe',  'text1':     '#0f1f1a',
    'text2':     '#374151',  'text3':     '#6b7280',
    'border':    '#d1d5db',  'surf2':     '#f8fafc',
}

def _h(k):
    from reportlab.lib import colors as _rc
    return _rc.HexColor(_C[k])

def _pdf_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib import colors as _rc
    N = getSampleStyleSheet()['Normal']
    def S(n, **kw): return ParagraphStyle(n, parent=N, **kw)
    return {
        'title':   S('csT',  fontSize=21, fontName='Helvetica-Bold',
                     textColor=_rc.white, leading=25, spaceAfter=2),
        'subtitle':S('csSub',fontSize=9,  textColor=_rc.HexColor('#a7f3d0'), leading=13),
        'h2':      S('csH2', fontSize=10, fontName='Helvetica-Bold',
                     textColor=_h('green_dk'), spaceAfter=0, spaceBefore=0),
        'body':    S('csBody',fontSize=9.5,textColor=_h('text2'),leading=15,spaceAfter=4),
        'small':   S('csSm', fontSize=7.5,textColor=_h('text3'),leading=11),
        'bullet':  S('csBul',fontSize=9,  textColor=_h('text2'),
                     leftIndent=10,spaceAfter=2,leading=13),
        'warn':    S('csW',  fontSize=8.5,textColor=_h('coral'),
                     backColor=_rc.HexColor(_C['coral_lt']),leading=13,
                     spaceAfter=5,leftIndent=8,rightIndent=8,borderPadding=5),
        'caution': S('csC',  fontSize=8.5,textColor=_h('amber'),
                     backColor=_rc.HexColor(_C['amber_lt']),leading=13,
                     spaceAfter=5,leftIndent=8,rightIndent=8,borderPadding=5),
        'ok':      S('csOk', fontSize=8.5,textColor=_h('green'),
                     backColor=_rc.HexColor(_C['green_lt']),leading=13,
                     spaceAfter=5,leftIndent=8,rightIndent=8,borderPadding=5),
        'th':      S('csTH', fontSize=9,  fontName='Helvetica-Bold',
                     textColor=_rc.white,leading=12),
        'td':      S('csTD', fontSize=9,  textColor=_h('text2'),leading=13),
        'td_sm':   S('csTDs',fontSize=8,  textColor=_h('text3'),leading=11),
        'td_bold': S('csTDb',fontSize=9,  fontName='Helvetica-Bold',
                     textColor=_h('green_dk'),leading=13),
        'disc':    S('csDis',fontSize=7.5,textColor=_h('text3'),
                     leading=11,alignment=TA_JUSTIFY),
    }

def _page_frame(canvas, doc):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as _rc
    PW, PH = A4
    canvas.saveState()
    canvas.setFillColor(_rc.HexColor(_C['green']))
    canvas.rect(0, PH-1.55*cm, PW, 1.55*cm, fill=1, stroke=0)
    canvas.setFillColor(_rc.white)
    canvas.setFont('Helvetica-Bold', 9.5)
    canvas.drawString(2*cm, PH-1.05*cm, 'CareSense  —  Medical Analysis Report')
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(_rc.HexColor('#a7f3d0'))
    canvas.drawRightString(PW-2*cm, PH-1.05*cm,
        f'Page {doc.page}  |  {datetime.datetime.now().strftime("%d %b %Y")}')
    canvas.setFillColor(_rc.HexColor(_C['border']))
    canvas.rect(0, 0, PW, 1.0*cm, fill=1, stroke=0)
    canvas.setFillColor(_rc.HexColor(_C['text3']))
    canvas.setFont('Helvetica', 6.5)
    canvas.drawString(2*cm, 0.38*cm,
        'AI-generated for educational use only — not a substitute for professional medical advice.')
    canvas.drawRightString(PW-2*cm, 0.38*cm, 'CareSense  ·  Ensemble ML  ·  Flask')
    canvas.restoreState()

def _sec(text, ST):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors as _rc
    cell = Table([[Paragraph(text, ST['h2'])]], colWidths=[16.5*cm])
    cell.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), _rc.HexColor(_C['green_xlt'])),
        ('LINEBEFORE',    (0,0),(0,-1),  3.5, _rc.HexColor(_C['green'])),
        ('TOPPADDING',    (0,0),(-1,-1), 5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 9),
        ('RIGHTPADDING',  (0,0),(-1,-1), 6),
    ]))
    return cell

def _ts(hcol, alt, extra=None):
    """Standard table style: green/violet header, alternating rows, word-wrap."""
    from reportlab.platypus import TableStyle
    from reportlab.lib import colors as _rc
    s = [
        ('BACKGROUND',     (0,0),(-1,0),  _rc.HexColor(hcol)),
        ('TEXTCOLOR',      (0,0),(-1,0),  _rc.white),
        ('FONTNAME',       (0,0),(-1,0),  'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [_rc.HexColor(alt), _rc.white]),
        ('GRID',           (0,0),(-1,-1), 0.4, _rc.HexColor(_C['border'])),
        ('FONTSIZE',       (0,0),(-1,-1), 9),
        ('TOPPADDING',     (0,0),(-1,-1), 6),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 6),
        ('LEFTPADDING',    (0,0),(-1,-1), 7),
        ('RIGHTPADDING',   (0,0),(-1,-1), 7),
        ('VALIGN',         (0,0),(-1,-1), 'TOP'),
        ('WORDWRAP',       (0,0),(-1,-1), 'CJK'),
    ]
    if extra:
        s.extend(extra)
    return TableStyle(s)

def P(text, style):
    """Convenience Paragraph factory."""
    from reportlab.platypus import Paragraph as _P
    return _P(str(text), style)

def _bullets(items, ST, cols=2):
    from reportlab.platypus import Table, TableStyle, Paragraph
    if not items: return None
    w = 16.5*cm / cols
    rows, row = [], []
    for item in items:
        row.append(Paragraph(f'• {item}', ST['bullet']))
        if len(row) == cols:
            rows.append(row); row = []
    if row:
        while len(row) < cols: row.append(Paragraph('', ST['bullet']))
        rows.append(row)
    t = Table(rows, colWidths=[w]*cols)
    t.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',   (0,0),(-1,-1), 2),
        ('RIGHTPADDING',  (0,0),(-1,-1), 4),
        ('TOPPADDING',    (0,0),(-1,-1), 1),
        ('BOTTOMPADDING', (0,0),(-1,-1), 1),
    ]))
    return t

def _chart_top5(top5, differential=None):
    diff  = differential or {}
    items = [(d, p, (diff.get(d) or {}).get('match_pct', 0) or 0)
             for d, p in top5.items()]
    items.sort(key=lambda x: -x[1])          # sort by probability
    names   = [x[0] for x in items][::-1]
    probs   = [x[1] for x in items][::-1]
    matches = [x[2] for x in items][::-1]
    n = len(names)
    G, A, R, GR = '#0a5c48','#b45309','#c0392b','#cbd5e1'
    fig, (ax1, ax2) = plt.subplots(1, 2,
        figsize=(9.5, max(2.8, n*0.72+0.8)),
        gridspec_kw={'width_ratios':[1.15,1]})
    fig.patch.set_facecolor('#f8fafc')
    for ax in (ax1, ax2):
        ax.set_facecolor('#f8fafc')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
    # Probability bars
    pcols = [G if p==max(probs) else GR for p in probs]
    b1 = ax1.barh(names, probs, color=pcols, height=0.52, edgecolor='white', linewidth=0.5)
    ax1.set_xlim(0, 118)
    ax1.set_xlabel('Probability (%)', fontsize=8, color='#6b7280')
    ax1.set_title('Ensemble Probability', fontsize=9, fontweight='bold', pad=6, color='#0f1f1a')
    ax1.tick_params(axis='y', labelsize=7.5, colors='#374151')
    ax1.tick_params(axis='x', labelsize=7,   colors='#6b7280')
    for bar, v in zip(b1, probs):
        ax1.text(bar.get_width()+1.2, bar.get_y()+bar.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=8, fontweight='bold',
                 color=G if v==max(probs) else '#6b7280')
    # Match bars
    mcols = [G if m>=75 else (A if m>=40 else R) for m in matches]
    b2 = ax2.barh(names, matches, color=mcols, height=0.52, edgecolor='white', linewidth=0.5)
    ax2.set_xlim(0, 118)
    ax2.set_xlabel('Symptom Match (%)', fontsize=8, color='#6b7280')
    ax2.set_title('Symptom Match', fontsize=9, fontweight='bold', pad=6, color='#0f1f1a')
    ax2.tick_params(axis='y', labelsize=7.5, colors='#374151')
    ax2.tick_params(axis='x', labelsize=7,   colors='#6b7280')
    for bar, v in zip(b2, matches):
        lc = G if v>=75 else (A if v>=40 else R)
        ax2.text(bar.get_width()+1.2, bar.get_y()+bar.get_height()/2,
                 f'{v}%', va='center', fontsize=8, fontweight='bold', color=lc)
    plt.tight_layout(pad=1.4)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=150, bbox_inches='tight', facecolor='#f8fafc')
    plt.close(); buf.seek(0)
    return buf

def _chart_gauges(sev_score, sev_label, confidence):
    G, A, R = '#0a5c48','#b45309','#c0392b'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 2.0),
                                    gridspec_kw={'width_ratios':[1.6,1]})
    fig.patch.set_facecolor('#f8fafc')
    for ax in (ax1, ax2):
        ax.set_facecolor('#f8fafc')
        for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#e5e7eb')
    sc = G if sev_score/35 < 0.35 else (A if sev_score/35 < 0.65 else R)
    ax1.barh([0],[35],       color='#e5e7eb',height=0.45,zorder=1)
    ax1.barh([0],[sev_score],color=sc,       height=0.45,zorder=2)
    ax1.set_xlim(0,40); ax1.set_yticks([])
    ax1.set_xlabel('Score (0–35)',fontsize=8,color='#6b7280')
    ax1.set_title(f'Severity: {sev_score} / 35  —  {sev_label}',
                   fontsize=9,fontweight='bold',pad=5,color='#0f1f1a')
    ax1.text(sev_score+0.5,0,str(sev_score),va='center',fontsize=11,fontweight='bold',color=sc)
    ax1.tick_params(axis='x',labelsize=7,colors='#6b7280')
    cc = G if confidence>=70 else (A if confidence>=40 else R)
    ax2.barh([0],[100],        color='#e5e7eb',height=0.45,zorder=1)
    ax2.barh([0],[confidence], color=cc,       height=0.45,zorder=2)
    ax2.set_xlim(0,118); ax2.set_yticks([])
    ax2.set_xlabel('Confidence (%)',fontsize=8,color='#6b7280')
    ax2.set_title(f'Confidence: {confidence:.1f}%',
                   fontsize=9,fontweight='bold',pad=5,color='#0f1f1a')
    ax2.text(confidence+1.5,0,f'{confidence:.1f}%',va='center',
             fontsize=11,fontweight='bold',color=cc)
    ax2.tick_params(axis='x',labelsize=7,colors='#6b7280')
    plt.tight_layout(pad=1.4)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=150, bbox_inches='tight', facecolor='#f8fafc')
    plt.close(); buf.seek(0)
    return buf

def generate_pdf(result: dict, buf: io.BytesIO):
    from reportlab.platypus import (SimpleDocTemplate, Spacer, Table, TableStyle,
                                     HRFlowable, Image as RLImage, KeepTogether)
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as _rc
    from reportlab.lib.units import cm

    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.1*cm, bottomMargin=1.55*cm)

    ST = _pdf_styles()
    S8 = Spacer(1, 8)
    S5 = Spacer(1, 5)

    info  = result.get('info', {})
    diff  = result.get('differential', {})
    top5  = result.get('top5', {})
    mvote = result.get('model_votes', {})
    dw    = result.get('duration_weights', {})
    rf    = result.get('risk_factors', {})
    syms  = result.get('selected_symptoms', [])
    DUR   = {'hours':'Hours','days':'Days','weeks':'Weeks','chronic':'Months+'}

    story = []

    # ── Cover banner ────────────────────────────────────────────
    cover = Table(
        [[P(f"<b>{result.get('disease','—')}</b>", ST['title'])],
         [P(f"Confidence: <b>{result.get('confidence',0):.1f}%</b>  ·  "
            f"Severity: <b>{result.get('severity_score',0)}</b> ({result.get('severity_label','—')})  ·  "
            f"Symptoms: <b>{result.get('sym_count',0)}</b>  ·  "
            f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}",
            ST['subtitle'])]],
        colWidths=[16.5*cm])
    cover.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), _rc.HexColor(_C['green'])),
        ('TOPPADDING',    (0,0),(-1,-1), 14),
        ('BOTTOMPADDING', (0,0),(-1,-1), 14),
        ('LEFTPADDING',   (0,0),(-1,-1), 16),
        ('RIGHTPADDING',  (0,0),(-1,-1), 12),
        ('LINEBELOW',     (0,-1),(-1,-1), 4, _rc.HexColor(_C['green_mid'])),
    ]))
    story += [cover, Spacer(1,10)]

    if result.get('rediagnosed'):
        story.append(P('↺  Re-analysed with added symptoms', ST['ok']))
    if result.get('validated'):
        story.append(P('✔  Result validated via follow-up', ST['ok']))
    if result.get('emergency'):
        story.append(P(
            f"🚨  EMERGENCY: {result.get('emergency_info',{}).get('message','Critical combination detected')}. "
            "Seek immediate medical attention.", ST['warn']))
    if result.get('consult_doctor'):
        story.append(P(
            f"⚠  Consultation advised — severity {result.get('severity_score',0)} "
            f"({result.get('severity_label','—')}). Confidence {result.get('confidence',0):.1f}% "
            f"is below the {result.get('threshold_used',35)}% threshold.", ST['caution']))
    story.append(S8)

    # ══ 1. PREDICTION SUMMARY ════════════════════════════════════
    story += [_sec('1  ·  Prediction Summary', ST), S5]
    consult_str = '⚠  Required' if result.get('consult_doctor') else '✔  Not Required'
    valid_str   = 'Yes — follow-up confirmed' if result.get('validated') else 'Initial prediction'
    sum_rows = [
        [P('<b>Field</b>',  ST['th']), P('<b>Value</b>', ST['th'])],
        [P('Predicted Disease',   ST['td_bold']), P(f"<b>{result.get('disease','—')}</b>", ST['td'])],
        [P('Ensemble Confidence', ST['td_bold']), P(f"{result.get('confidence',0):.2f}%", ST['td'])],
        [P('Severity Score',      ST['td_bold']), P(f"{result.get('severity_score',0)} / 35  —  {result.get('severity_label','—')}", ST['td'])],
        [P('Dynamic Threshold',   ST['td_bold']), P(f"{result.get('threshold_used',35)}% confidence required", ST['td'])],
        [P('Symptoms Analysed',   ST['td_bold']), P(str(result.get('sym_count', 0)), ST['td'])],
        [P('Doctor Consultation', ST['td_bold']), P(consult_str, ST['td'])],
        [P('Validated',           ST['td_bold']), P(valid_str,   ST['td'])],
    ]
    if rf:
        parts = [rf['age_group']] if rf.get('age_group') else []
        for k, lbl in [('diabetic','Diabetic'),('hypertensive','Hypertensive'),('smoker','Smoker')]:
            if rf.get(k): parts.append(lbl)
        if parts:
            sum_rows.append([P('Risk Profile', ST['td_bold']),
                              P(', '.join(parts), ST['td'])])
    st_tbl = Table(sum_rows, colWidths=[5.2*cm, 11.3*cm])
    st_tbl.setStyle(_ts(_C['green'], _C['green_xlt']))
    story += [KeepTogether(st_tbl), S8]

    # ══ 2. ENSEMBLE MODEL VOTES ══════════════════════════════════
    if mvote:
        story += [_sec('2  ·  Ensemble Model Votes', ST), S5]
        W = {'Random Forest':'50%','SVM':'35%','Naive Bayes':'15%'}
        primary = result.get('disease','')
        v_rows = [[P(h, ST['th']) for h in
                   ['Model','Weight','Predicted Disease','Conf.','Agrees?']]]
        for mname, vote in mvote.items():
            agrees = vote.get('disease') == primary
            v_rows.append([
                P(mname,                           ST['td']),
                P(W.get(mname,'—'),                ST['td']),
                P(vote.get('disease','—'),         ST['td']),
                P(f"{vote.get('confidence',0):.1f}%", ST['td']),
                P('<b>✔ Yes</b>' if agrees else '✗ No', ST['td']),
            ])
        v_tbl = Table(v_rows, colWidths=[3.8*cm, 1.7*cm, 5.5*cm, 2.2*cm, 3.3*cm])
        v_tbl.setStyle(_ts(_C['green'], _C['green_xlt']))
        story += [KeepTogether(v_tbl), S8]

    # ══ 3. DIFFERENTIAL DIAGNOSIS CHART ══════════════════════════
    story += [_sec('3  ·  Differential Diagnosis', ST), S5]
    if top5:
        n = len(top5)
        story.append(RLImage(_chart_top5(top5, diff),
                              width=16.5*cm, height=max(4.2*cm, n*1.05*cm+1.5*cm)))
    story.append(S8)

    # ══ 4. SYMPTOM MATCH BREAKDOWN ═══════════════════════════════
    if diff:
        story += [_sec('4  ·  Symptom Match Analysis', ST), S5]
        sorted_d = sorted(diff.keys(), key=lambda d: -(diff[d].get('match_pct',0) or 0))
        def _fmt(lst):
            return ', '.join(s.replace('_',' ') for s in lst) if lst else '—'
        m_rows = [[P(h, ST['th']) for h in
                   ['Disease','Match %','Strong Symptoms','Moderate Symptoms','Also Typical']]]
        for d in sorted_d:
            e  = diff[d]
            mp = e.get('match_pct', 0) or 0
            m_rows.append([
                P(f'<b>{d}</b>',           ST['td']),
                P(f'<b>{mp}%</b>',         ST['td']),
                P(_fmt(e.get('strong',  [])), ST['td_sm']),
                P(_fmt(e.get('moderate',[])), ST['td_sm']),
                P(_fmt(e.get('missing', [])), ST['td_sm']),
            ])
        # Wider columns so symptom lists wrap properly
        m_tbl = Table(m_rows, colWidths=[4.0*cm, 1.6*cm, 3.5*cm, 3.5*cm, 3.9*cm])
        m_tbl.setStyle(_ts(_C['green'], _C['green_xlt']))
        story += [KeepTogether(m_tbl), S8]

    # ══ 5. SEVERITY & CONFIDENCE GAUGES ══════════════════════════
    story += [_sec('5  ·  Severity & Confidence', ST), S5]
    story.append(RLImage(
        _chart_gauges(result.get('severity_score',0),
                      result.get('severity_label','Unknown'),
                      result.get('confidence',0)),
        width=16.5*cm, height=3.0*cm))
    story.append(S8)

    # ══ 6. REPORTED SYMPTOMS ═════════════════════════════════════
    story += [_sec('6  ·  Reported Symptoms', ST), S5]
    if syms:
        padded = syms if len(syms)%2==0 else syms+['']
        s_rows = [[P(h, ST['th']) for h in ['Symptom','Duration','Symptom','Duration']]]
        for i in range(0, len(padded), 2):
            s1, s2 = padded[i], padded[i+1]
            d1 = DUR.get(dw.get(s1,''), dw.get(s1,'') or '—')
            d2 = DUR.get(dw.get(s2,''), dw.get(s2,'') or '—') if s2 else ''
            s_rows.append([
                P(s1.replace('_',' ').title(), ST['td']),
                P(d1, ST['td_sm']),
                P(s2.replace('_',' ').title() if s2 else '', ST['td']),
                P(d2 if s2 else '', ST['td_sm']),
            ])
        s_tbl = Table(s_rows, colWidths=[5.5*cm, 2.5*cm, 5.5*cm, 3.0*cm])
        s_tbl.setStyle(_ts(_C['green'], _C['green_xlt'], extra=[
            ('LINEAFTER',(1,0),(1,-1),1.5,_rc.HexColor(_C['border']))]))
        story += [KeepTogether(s_tbl)]
    else:
        story.append(P('No symptoms recorded.', ST['body']))
    story.append(S8)

    # ══ 7. ABOUT THIS DISEASE ════════════════════════════════════
    story += [_sec('7  ·  About This Disease', ST), S5]
    story.append(P(info.get('description','No description available.'), ST['body']))
    story.append(S8)

    # ══ 8. MEDICATIONS ═══════════════════════════════════════════
    story += [_sec('8  ·  Recommended Medications', ST), S5]
    story.append(P('⚠  General information only. Consult a licensed physician for dosage and suitability.',
                   ST['small']))
    story.append(Spacer(1,4))
    meds = info.get('medications', [])
    if meds:
        med_rows = [[P(h, ST['th']) for h in ['Medication','Note']]]
        for m in meds:
            med_rows.append([P(str(m), ST['td']),
                              P('Dosage per physician', ST['td_sm'])])
        med_tbl = Table(med_rows, colWidths=[6*cm, 10.5*cm])
        med_tbl.setStyle(_ts(_C['violet'], _C['violet_lt']))
        story.append(KeepTogether(med_tbl))
    else:
        story.append(P('No medication data available.', ST['body']))
    story.append(S8)

    # ══ 9. PRECAUTIONS ═══════════════════════════════════════════
    story += [_sec('9  ·  Precautions', ST), S5]
    precs = info.get('precautions', [])
    if precs:
        g = _bullets(precs, ST)
        if g: story.append(g)
    else:
        story.append(P('No precaution data available.', ST['body']))
    story.append(S8)

    # ══ 10. DIET ═════════════════════════════════════════════════
    story += [_sec('10  ·  Recommended Diet', ST), S5]
    diets = info.get('diets', [])
    if diets:
        g = _bullets(diets, ST)
        if g: story.append(g)
    else:
        story.append(P('No diet data available.', ST['body']))
    story.append(S8)

    # ══ 11. WORKOUT ══════════════════════════════════════════════
    story += [_sec('11  ·  Lifestyle & Workout', ST), S5]
    wouts = info.get('workouts', [])
    if wouts:
        g = _bullets(wouts[:10], ST)
        if g: story.append(g)
    else:
        story.append(P('No workout data available.', ST['body']))
    story.append(S8)

    # ══ 12. HOW IT WORKS ═════════════════════════════════════════
    story += [_sec('12  ·  How CareSense Works', ST), S5]
    how_rows = [
        [P('<b>Component</b>', ST['th']), P('<b>Explanation</b>', ST['th'])],
        [P('Ensemble Voting',    ST['td_bold']),
         P('Random Forest (50%) + SVM (35%) + Naive Bayes (15%) each produce probability '
           'vectors over all 41 diseases. A weighted average gives the final prediction.',
           ST['td'])],
        [P('Severity Weighting', ST['td_bold']),
         P('Each symptom is multiplied by its clinical weight (1–7) before the feature '
           'vector is fed to the models — chest pain counts far more than mild itching.',
           ST['td'])],
        [P('Dynamic Threshold',  ST['td_bold']),
         P(f'Severity score {result.get("severity_score",0)} ({result.get("severity_label","—")}) '
           f'required {result.get("threshold_used",35)}% confidence. '
           'Higher severity triggers a stricter threshold and consultation advisory.',
           ST['td'])],
        [P('Differential Diagnosis', ST['td_bold']),
         P('Training frequency data determines match strength: ≥70% = strong, '
           '30–69% = moderate, <30% = weak. Missing typical symptoms shown separately.',
           ST['td'])],
    ]
    h_tbl = Table(how_rows, colWidths=[4.2*cm, 12.3*cm])
    h_tbl.setStyle(_ts(_C['green'], _C['green_xlt']))
    story += [KeepTogether(h_tbl), Spacer(1,16)]

    story.append(HRFlowable(width='100%', thickness=0.8,
                              color=_rc.HexColor(_C['border']), spaceAfter=6))
    story.append(P(
        '<b>DISCLAIMER:</b> CareSense is an AI-based educational tool for academic purposes only. '
        'This report does NOT constitute medical advice, diagnosis, or treatment. '
        'Always consult a qualified physician for any medical condition. '
        'Never disregard professional medical advice because of information from this system. '
        'In an emergency call your local emergency services immediately.',
        ST['disc']))

    doc.build(story, onFirstPage=_page_frame, onLaterPages=_page_frame)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
