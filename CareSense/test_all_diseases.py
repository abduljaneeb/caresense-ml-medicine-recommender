import io
import traceback
from app import app, generate_pdf, le, get_disease_info, key_sym_map

client = app.test_client()

failures = 0
for disease in le.classes_:
    info = get_disease_info(disease)
    result = {
        'disease': disease,
        'confidence': 95.0,
        'severity_score': 20,
        'severity_label': 'High',
        'sym_count': 5,
        'info': info,
        'differential': {
             disease: {'match_pct': 100, 'strong': key_sym_map.get(disease, [])[:3], 'moderate': [], 'missing': []}
        },
        'top5': {disease: 95.0, 'Other': 5.0},
        'model_votes': {},
        'duration_weights': {},
        'risk_factors': {},
        'selected_symptoms': key_sym_map.get(disease, [])[:5]
    }
    buf = io.BytesIO()
    try:
        generate_pdf(result, buf)
    except Exception as e:
        failures += 1
        print(f"FAILED on disease: {disease}")
        traceback.print_exc()

print(f"Total failures: {failures} out of {len(le.classes_)}")
