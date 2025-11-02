# ALLOH_23CG034037_EMOTION_DETECTION_WEB_APP

A Flask web app for detecting human emotions from images and live webcam snapshots.  
Stores user name, image path, and predicted emotion in a SQLite database.

## Structure
- `app.py` – Flask backend
- `model.py` – Keras training script (saves `models/emotion_cnn.h5`)
- `templates/` – HTML files
- `static/` – CSS (aesthetic color scheme)
- `uploads/` – runtime image storage
- `models/` – trained models
- `aurora_emote_v1.pkl` – creative-named model at root (dummy for bootstrapping)
- `emotrack.db` – SQLite DB
- `Requirements.txt` – dependencies
- `link_to_my_web_app.txt` – hosting URL format

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r Requirements.txt
python app.py
# visit http://127.0.0.1:5000
```

## Train
Update `load_data()` in `model.py` with real data, then:
```bash
python model.py
```
This creates `models/emotion_cnn.h5`. The app auto-loads it.

## Deploy (Render)
1. Push to GitHub.
2. On render.com → New Web Service → select repo.
3. Build: `pip install -r Requirements.txt`
4. Start: `gunicorn app:app`
5. Add `SECRET_KEY` env var (optional).
6. Put the live URL into `link_to_my_web_app.txt` using the format `Render – https://...`
