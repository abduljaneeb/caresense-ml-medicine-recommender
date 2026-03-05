from app import app
import json
import traceback

client = app.test_client()

payload = {
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
    response = client.post('/download_pdf', json=payload)
    print("Status code:", response.status_code)
    if response.status_code != 200:
        print("Response text:", response.get_data(as_text=True))
    else:
        print("PDF downloaded via Flask client successfully.")
except Exception as e:
    print("Exception during request:")
    traceback.print_exc()
