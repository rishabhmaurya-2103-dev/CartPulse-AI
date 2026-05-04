# 🛒 CartSense — Cart Abandonment Predictor

A full-stack web application that predicts whether a shopping cart will be **abandoned or purchased** using a real trained **Random Forest machine learning model**.

Built with Flask (Python backend) + vanilla HTML/CSS/JS frontend with live Chart.js visualizations.

---

## 🖥️ Screenshots

| Dashboard | Predict | History |
|-----------|---------|---------|
| Feature importance, price sensitivity & session charts | Real-time ML prediction with risk level | Full prediction log with line & pie charts |

---

## ✨ Features

- 🔮 **Real ML Inference** — Every prediction runs through the actual trained `RandomForestClassifier` (100 trees, depth 12)
- 📊 **Live Charts** — Feature importance, price sensitivity curve, session depth analysis, outcome distribution
- 📋 **Prediction History** — Full session log with charts showing trends over time
- ⚡ **4 Pages** — Dashboard · Predict · History · About
- 🎨 **Dark UI** — Premium dark theme with animated components

---

## 🧠 Model Info

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees |
| Max Depth | 12 |
| Accuracy | 79.4% |
| ROC-AUC | 0.871 |
| Training Samples | 11,966 |
| Features | 10 |

### Features Used
`price` · `brand` · `category_code` · `total_events` · `view_count` · `cart_count` · `unique_products` · `avg_price` · `max_price` · `min_price`

---

## 📦 Dataset

**Kaggle:** [ecommerce-behavior-data-from-multi-category-store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

- 1 million rows sampled from October–November 2019
- Cart events extracted and session-level features aggregated
- 176 unique brands · 69 product categories

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/cartsense.git
cd cartsense
```

### 2. Install dependencies
```bash
pip install flask joblib scikit-learn numpy
```

### 3. Add the model files
Download or copy these 3 files into the **same folder** as `app.py`:
```
cartsense/
├── app.py
├── index.html
├── cart_abandonment_model.pkl   ← add this
├── brand_encoder.pkl            ← add this
└── category_encoder.pkl         ← add this
```

> The `.pkl` files are not included in this repo due to size. You can generate them by running the training notebook, or get them from the project author.

### 4. Run the server
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:7860
```

---

## 📁 Project Structure

```
cartsense/
├── app.py          # Flask backend — loads models, exposes API endpoints
├── index.html      # Full frontend — 4-page SPA with charts
└── README.md
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend |
| `/api/meta` | GET | Returns brands, categories, model info |
| `/api/predict` | POST | Runs a single prediction |
| `/api/batch_predict` | POST | Price sensitivity sweep (9 price points) |
| `/api/history` | GET | Returns session prediction log |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn RandomForestClassifier |
| Backend | Flask (Python) |
| Frontend | HTML · CSS · JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Syne · DM Mono · DM Sans |

---

## 📓 Training Notebook

The model was trained in Google Colab using the Kaggle dataset. The pipeline:

1. Load raw e-commerce CSV (1M rows)
2. Filter `cart` events only
3. Aggregate session-level features per user session
4. Label encode `brand` and `category_code`
5. Train/test split (80/20, stratified)
6. Fit `RandomForestClassifier` with balanced class weights
7. Export model + encoders as `.pkl` files via `joblib`

---

## 👤 Author

Made by **Rishabh Maurya**  
📧 rishabhmaurya2103@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rishabh-maurya-a33754374/) · [Kaggle](https://kaggle.com)

---

## 📄 License

MIT License — free to use, modify and distribute.
