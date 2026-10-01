from pathlib import Path
import json, joblib, numpy as np
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts/breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts/metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

app = FastAPI(
    title="LIMS & R&D Botanical API",
    version=metadata["model_version"],
    description="Hệ thống QC Nhận diện và Phân tích"
)

templates = Jinja2Templates(directory="app/templates")

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Exactly 30 named numeric features")

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str

def build_vector(payload: PredictionRequest) -> np.ndarray:
    expected = metadata["feature_names"]
    received = set(payload.features)
    if sorted(set(expected) - received) or sorted(received - set(expected)):
        raise HTTPException(status_code=422, detail="Sai cấu trúc đặc trưng")
    return np.array([[payload.features[name] for name in expected]], dtype=float)

@app.get("/login")
def login_page(request: Request, error: int = 0):
    # Đã sửa lại cú pháp chuẩn cho FastAPI mới nhất
    return templates.TemplateResponse(
        request=request, 
        name="login.html", 
        context={"error": error}
    )

@app.post("/login")
def login_process(username: str = Form(...), password: str = Form(...)):
    if username == "bin" and password == "1234":
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="lims_session", value="authenticated", httponly=True)
        return response
    return RedirectResponse(url="/login?error=1", status_code=303)

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("lims_session")
    return response

@app.get("/")
def root(request: Request):
    if not request.cookies.get("lims_session"):
        return RedirectResponse(url="/login")
        
    # Đã sửa lại cú pháp chuẩn cho FastAPI mới nhất
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "feature_names": metadata["feature_names"],
            "warning": metadata["warning"]
        }
    )

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/metadata")
def get_metadata(): return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_vector(payload)
    predicted_class = int(model.predict(x)[0])
    p = {int(c): float(v) for c, v in zip(list(model.named_steps["svc"].classes_), model.predict_proba(x)[0])}
    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][str(predicted_class)],
        probability_malignant=p[0], probability_benign=p[1],
        model_version=metadata["model_version"], warning=metadata["warning"]
    )
