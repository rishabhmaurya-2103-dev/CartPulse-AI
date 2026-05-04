import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import json
import os
import time
from datetime import datetime

app = Flask(__name__)

# ── Load real ML models ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model        = joblib.load(os.path.join(BASE, 'cart_abandonment_model.pkl'))
brand_enc    = joblib.load(os.path.join(BASE, 'brand_encoder.pkl'))
category_enc = joblib.load(os.path.join(BASE, 'category_encoder.pkl'))

BRANDS     = sorted(list(brand_enc.classes_))
CATEGORIES = sorted(list(category_enc.classes_))

# Features order (from notebook):
# price, brand, category_code, total_events, view_count,
# cart_count, unique_products, avg_price, max_price, min_price
FEATURE_NAMES = [
    'price','brand','category_code','total_events','view_count',
    'cart_count','unique_products','avg_price','max_price','min_price'
]
FEATURE_IMPORTANCES = model.feature_importances_.tolist()

# ── In-memory history store ──────────────────────────────────────────────────
history = []

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

@app.route('/api/meta')
def meta():
    return jsonify({
        'brands': BRANDS,
        'categories': CATEGORIES,
        'feature_names': FEATURE_NAMES,
        'feature_importances': FEATURE_IMPORTANCES,
        'model_info': {
            'type': 'RandomForestClassifier',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'n_features': model.n_features_in_,
            'training_samples': 11966,
            'test_samples': 2992,
            'accuracy': 0.7941,
            'roc_auc': 0.8712,
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        brand_raw = data.get('brand', 'unknown').lower()
        cat_raw   = data.get('category_code', 'unknown')

        # Encode - fallback to 'unknown' if unseen label
        try:
            brand_code = int(brand_enc.transform([brand_raw])[0])
        except Exception:
            brand_code = int(brand_enc.transform(['unknown'])[0])

        try:
            cat_code = int(category_enc.transform([cat_raw])[0])
        except Exception:
            cat_code = int(category_enc.transform(['unknown'])[0])

        price           = float(data.get('price', 100))
        total_events    = int(data.get('total_events', 5))
        view_count      = int(data.get('view_count', 4))
        cart_count      = int(data.get('cart_count', 1))
        unique_products = int(data.get('unique_products', 1))
        avg_price       = float(data.get('avg_price', price))
        max_price       = float(data.get('max_price', price))
        min_price       = float(data.get('min_price', price))

        X = np.array([[price, brand_code, cat_code, total_events,
                       view_count, cart_count, unique_products,
                       avg_price, max_price, min_price]])

        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()

        abandon_prob  = round(proba[1] * 100, 1)
        purchase_prob = round(proba[0] * 100, 1)

        risk = 'HIGH' if abandon_prob >= 65 else ('MEDIUM' if abandon_prob >= 40 else 'LOW')

        result = {
            'prediction': pred,
            'abandoned': bool(pred == 1),
            'abandon_probability': abandon_prob,
            'purchase_probability': purchase_prob,
            'risk_level': risk,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'input': {
                'brand': brand_raw,
                'category': cat_raw,
                'price': price,
                'total_events': total_events,
                'view_count': view_count,
                'cart_count': cart_count,
                'unique_products': unique_products,
            }
        }

        # Save to history (keep last 100)
        history.append(result)
        if len(history) > 100:
            history.pop(0)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/history')
def get_history():
    return jsonify(history[-50:])

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Run multiple predictions with varying price to show sensitivity curve."""
    data = request.get_json()
    base = data.copy()
    results = []

    brand_raw = base.get('brand', 'unknown').lower()
    cat_raw   = base.get('category_code', 'unknown')

    try:
        brand_code = int(brand_enc.transform([brand_raw])[0])
    except:
        brand_code = int(brand_enc.transform(['unknown'])[0])
    try:
        cat_code = int(category_enc.transform([cat_raw])[0])
    except:
        cat_code = int(category_enc.transform(['unknown'])[0])

    # Sweep price from 10% to 300% of given price
    base_price = float(base.get('price', 100))
    for factor in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        p = round(base_price * factor, 2)
        X = np.array([[p, brand_code, cat_code,
                       int(base.get('total_events', 5)),
                       int(base.get('view_count', 4)),
                       int(base.get('cart_count', 1)),
                       int(base.get('unique_products', 1)),
                       p, p, p]])
        proba = model.predict_proba(X)[0]
        results.append({'price': p, 'abandon_prob': round(proba[1]*100, 1)})

    return jsonify(results)

if __name__ == '__main__':
    print("✅ Model loaded — RandomForest with", model.n_estimators, "trees")
    print("✅ Brands:", len(BRANDS), "| Categories:", len(CATEGORIES))
    app.run(debug=False, port=7860)
