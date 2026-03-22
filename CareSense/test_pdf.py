import io
import traceback
from app import generate_pdf

result = {
    'disease': 'Diabetes',
    'confidence': 85.5,
    'severity_score': 15,
    'severity_label': 'Moderate',
    'sym_count': 3,
    'info': {
        'description': 'Test descriptoon',
        'medications': ['Med1'],
        'precautions': ['Prec1'],
        'diets': ['Diet1'],
        'workouts': ['Workout1']
    },
    'differential': {
        'Diabetes': {'match_pct': 80, 'strong': ['thirst'], 'moderate': [], 'missing': []}
    },
    'top5': {'Diabetes': 85.5, 'Hypertension': 10.0},
    'model_votes': {},
    'duration_weights': {},
    'risk_factors': {},
    'selected_symptoms': ['thirst', 'fatigue']
}

try:
    buf = io.BytesIO()
    generate_pdf(result, buf)
    buf.seek(0)
    with open('test_output.pdf', 'wb') as f:
        f.write(buf.read())
    print("PDF generated successfully.")
except Exception as e:
    print("PDF generation failed:")
    traceback.print_exc()
