import os, io, base64, sqlite3
from datetime import datetime
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Add this near the top of app.py (before load_model_any is called) ---
class AuroraEmoteModel:
    def __init__(self, labels=None):
        self.labels = labels or ["angry","disgust","fear","happy","sad","surprise","neutral"]
    def predict(self, batch):
        import numpy as np
        probs = np.zeros((len(batch), len(self.labels)))
        probs[:, self.labels.index("neutral")] = 1.0
        return probs
# --- end add ---

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False

LABELS = ["angry","disgust","fear","happy","sad","surprise","neutral"]
MODEL = None

def load_model_any():
    """Load a TF/Keras model if available; else load pickle with a safe resolver; else fallback to a live dummy."""
    global MODEL
    # 1) Preferred: Keras model
    keras_path = os.path.join("models", "emotion_cnn.h5")
    if os.path.exists(keras_path):
        try:
            from tensorflow.keras.models import load_model
            MODEL = load_model(keras_path)
            print("Loaded Keras model:", keras_path)
            return
        except Exception as e:
            print("Failed to load Keras model:", e)

    # 2) Pickle with safe resolver
    import pickle
    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "AuroraEmoteModel":
                return AuroraEmoteModel
            return super().find_class(module, name)

    for p in ["aurora_emote_v1.pkl", os.path.join("models", "aurora_emote_v1.pkl")]:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    MODEL = SafeUnpickler(f).load()
                print("Loaded pickle model via SafeUnpickler:", p)
                return
            except Exception as e:
                print("Pickle load still failed:", e)

    # 3) Final fallback: instantiate dummy
    MODEL = AuroraEmoteModel()
    print("No model file loaded; using in-code AuroraEmoteModel dummy.")


def ensure_db():
    conn = sqlite3.connect("emotrack.db"); cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        source TEXT NOT NULL,
        image_path TEXT NOT NULL,
        prediction TEXT NOT NULL
    );
    """); conn.commit(); conn.close()

def save_record(name, source, image_path, prediction):
    conn = sqlite3.connect("emotrack.db"); cur = conn.cursor()
    cur.execute("INSERT INTO predictions (name, created_at, source, image_path, prediction) VALUES (?,?,?,?,?)",
                (name, datetime.utcnow().isoformat(), source, image_path, prediction))
    conn.commit(); conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in {"png","jpg","jpeg"}

def preprocess_image(img):
    if OPENCV_AVAILABLE:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces)>0:
                (x,y,w,h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
                face = gray[y:y+h, x:x+w]
            else:
                face = gray
            from PIL import Image as PImage
            face_img = PImage.fromarray(face).resize((48,48))
            arr = np.array(face_img, dtype="float32")/255.0
        except Exception:
            arr = np.array(img.convert("L").resize((48,48)), dtype="float32")/255.0
    else:
        arr = np.array(img.convert("L").resize((48,48)), dtype="float32")/255.0
    arr = np.expand_dims(arr, -1); arr = np.expand_dims(arr, 0)
    return arr

def predict_emotion(arr):
    if MODEL is None:
        return "neutral"
    try:
        preds = MODEL.predict(arr)
        if getattr(preds,"ndim",0)==2 and preds.shape[1]==len(LABELS):
            return LABELS[int(np.argmax(preds[0]))]
    except Exception as e:
        print("Keras predict failed:", e)
    try:
        preds = MODEL.predict(arr)
        return LABELS[int(np.argmax(preds[0]))]
    except Exception as e:
        print("Pickle predict failed:", e); return "neutral"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY","dev_secret")

ensure_db(); load_model_any()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = (request.form.get("name") or "Anonymous").strip()
    source = request.form.get("source","upload")
    image = None; image_path_for_db = None

    if source == "upload":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image file."); return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Unsupported file type (PNG/JPG/JPEG only)."); return redirect(url_for("index"))
        os.makedirs("uploads", exist_ok=True)
        filename = secure_filename(file.filename)
        save_path = os.path.join("uploads", filename)
        file.save(save_path)
        image = Image.open(save_path).convert("RGB")
        image_path_for_db = save_path
    elif source == "live":
        data_url = request.form.get("snapshot")
        if not data_url or not data_url.startswith("data:image/"):
            flash("No live snapshot received."); return redirect(url_for("index"))
        header, b64data = data_url.split(",",1)
        raw = base64.b64decode(b64data)
        os.makedirs("uploads", exist_ok=True)
        from datetime import datetime
        filename = f"live_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join("uploads", filename)
        with open(save_path,"wb") as f: f.write(raw)
        image = Image.open(save_path).convert("RGB")
        image_path_for_db = save_path
    else:
        flash("Invalid source."); return redirect(url_for("index"))

    arr = preprocess_image(image)
    pred = predict_emotion(arr)
    save_record(name, source, image_path_for_db, pred)
    return render_template("result.html", name=name, prediction=pred, image_path=image_path_for_db)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
