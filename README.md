# 💵 NoteGuard — Fake vs Real Bank Note Detector

A Machine Learning web app that detects whether a bank note is **Real or Fake** using wavelet-transformed image features.

Built with **Flask**, **FastAPI**, and **TensorFlow**.

---

## 🖼️ Preview

| Real Note ✅ | Fake Note ❌ |
|---|---|
| img.png | img_1.png |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
https://github.com/diveshkumar2233/fake-note-vs-real-note
cd fake-note-vs-real-note
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your model files
Place these inside the `models/` folder:
```
models/
├── model.h5
└── scaler.pkl
```

### 5. Run the app

**Flask** (port 5000):
```bash
python app.py
```

**FastAPI** (port 8000):
```bash
python fastapi_app.py
```

---

## 🌐 URLs

| Server | URL | Docs |
|---|---|---|
| Flask | http://localhost:5000 | — |
| FastAPI | http://localhost:8000 | http://localhost:8000/docs |

---

## 📁 Project Structure

```
FAKE NOTE VS REAL NOTE/
├── .env.example           ← copy to .env and edit
├── .gitignore
├── app.py                 ← Flask backend
├── fastapi_app.py         ← FastAPI backend
├── requirements.txt
├── setup.bat              ← Windows one-click run
├── setup.sh               ← Mac/Linux one-click run
├── models/
│   ├── model.h5           ← Keras model (not tracked by git)
│   └── scaler.pkl         ← scaler (not tracked by git)
├── static/
│   ├── img.png            ← Real note sample
│   └── img_1.png          ← Fake note sample
└── templates/
    └── index.html         ← Frontend UI
```

---

## 🧠 Features Used

| Feature | Description |
|---|---|
| Variance | Variance of wavelet transformed image |
| Skewness | Skewness of wavelet transformed image |
| Curtosis | Curtosis of wavelet transformed image |
| Entropy | Entropy of image |

---

## 🛠️ Tech Stack

- **Frontend** → HTML, CSS, python 
- **Backend** → Flask + FastAPI
- **ML Model** → TensorFlow / Keras
- **Preprocessing** → scikit-learn (StandardScaler)
- **Dataset** → [Banknote Authentication — UCI](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and update:

```env
FLASK_APP=app.py
FLASK_DEBUG=1
FLASK_RUN_PORT=5000
FASTAPI_PORT=8000
SECRET_KEY=your-secret-key-here
MODEL_PATH=models/model.h5
SCALER_PATH=models/scaler.pkl
```

---

## 📄 License

MIT License — free to use and modify.
