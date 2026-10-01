from pathlib import Path
import json, joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Nạp đường dẫn và mô hình từ thư mục artifacts
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts/breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts/metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only"
)

# 2. Khai báo cấu trúc dữ liệu đầu vào và đầu ra
class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Exactly 30 named numeric features")

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str

# 3. Hàm kiểm tra nghiêm ngặt 30 đặc trưng đầu vào
def build_vector(payload: PredictionRequest) -> np.ndarray:
    expected = metadata["feature_names"]
    received = set(payload.features)
    missing = sorted(set(expected) - received)
    extra = sorted(received - set(expected))
    
    if missing or extra:
        raise HTTPException(status_code=422, detail={"missing": missing, "extra": extra})
        
    values = np.array([[payload.features[name] for name in expected]], dtype=float)
    if not np.isfinite(values).all():
        raise HTTPException(422, "Features must be finite numbers")
    return values

# 4. Các đường dẫn (endpoints) của API
@app.get("/")
def root():
    return {"service": "Breast Cancer SVM API", "docs": "/docs", "warning": metadata["warning"]}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_version": metadata["model_version"]}

@app.get("/metadata")
def get_metadata():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_vector(payload)
    predicted_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    classes = list(model.named_steps["svc"].classes_)
    p = {int(c): float(v) for c, v in zip(classes, probabilities)}
    
    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][str(predicted_class)],
        probability_malignant=p[0],
        probability_benign=p[1],
        model_version=metadata["model_version"],
        warning=metadata["warning"]
    )