import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

from nids_pipeline import (
    train_model_from_csv,
    load_model,
    predict_from_csv,
    ensure_directories,
)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

ALLOWED_EXTENSIONS = {"csv"}

ensure_directories([UPLOAD_FOLDER, MODELS_FOLDER, DATA_FOLDER])


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    model_info = None
    model_path = os.path.join(MODELS_FOLDER, "model.joblib")
    if os.path.exists(model_path):
        model_info = {
            "path": model_path,
            "updated_at": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    return render_template("index.html", model_info=model_info)


@app.route("/train", methods=["POST"]) 
def train():
    csv_name = request.form.get("train_csv_name", "").strip()
    label_column = request.form.get("label_column", "label").strip() or "label"
    if not csv_name:
        flash("Please provide the training CSV file name located in the data/ folder.", "error")
        return redirect(url_for("index"))

    csv_path = os.path.join(DATA_FOLDER, csv_name)
    if not os.path.exists(csv_path):
        flash(f"Training file not found: data/{csv_name}", "error")
        return redirect(url_for("index"))

    try:
        metrics = train_model_from_csv(csv_path=csv_path, label_column=label_column, models_dir=MODELS_FOLDER)
        flash(
            f"Training complete. Accuracy: {metrics.get('accuracy'):.4f}, F1: {metrics.get('f1'):.4f}",
            "success",
        )
    except Exception as exc:
        flash(f"Training failed: {exc}", "error")
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"]) 
def predict():
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only CSV files are allowed.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        model_bundle = load_model(models_dir=MODELS_FOLDER)
        if model_bundle is None:
            flash("Model not found. Train a model first.", "error")
            return redirect(url_for("index"))

        predictions_df, summary = predict_from_csv(model_bundle, save_path)

        # Save predictions to a timestamped CSV in uploads
        out_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        predictions_df.to_csv(out_path, index=False)

        flash(
            f"Predictions complete. {summary.get('num_rows', 0)} rows processed. Download available.",
            "success",
        )
        return send_file(out_path, as_attachment=True)
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


