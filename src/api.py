from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import manager

app = FastAPI(title="Cancer Prediction API")

class TrainConfig(BaseModel):
    epochs: int = 50
    lr: float = 0.01
    hidden_dim: int = 32

class DatasetName(BaseModel):
    name: str

class PredictFeatures(BaseModel):
    features: list[float]

@app.get("/datasets")
def get_datasets():
    return {"available_datasets": manager.get_available_datasets()}

@app.post("/load-dataset")
def load_dataset(data: DatasetName):
    if data.name not in manager.get_available_datasets():
        raise HTTPException(400, "Dataset not found")
    info = manager.load_data(data.name)
    return {"status": "loaded", "data": info}

@app.get("/info")
def get_info():
    return {"status": "loaded", "data": {"features": manager.feature_names, "train_size": len(manager.y_train), "target_names": manager.target_names}}

@app.get("/plot/pca")
def get_pca():
    return {"image_base64": manager.get_pca_plot()}

@app.post("/train")
def train_model(config: TrainConfig):
    acc, cm = manager.train(epochs=config.epochs, lr=config.lr, hidden_dim=config.hidden_dim)
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "training_plot_base64": manager.get_training_plot(),
        "confusion_plot_base64": manager.get_confusion_plot(cm),
        "decision_boundary_base64": manager.get_decision_boundary_plot()
    }

@app.get("/plot/decision-boundary")
def get_decision_boundary():
    if not manager.is_trained: raise HTTPException(400, "Model not trained yet")
    plot_b64 = manager.get_decision_boundary_plot()
    if not plot_b64: raise HTTPException(500, "Failed to generate boundary plot")
    return {"image_base64": plot_b64}

@app.post("/predict")
def predict(data: PredictFeatures):
    if len(data.features) != len(manager.feature_names): raise HTTPException(400, f"Expected {len(manager.feature_names)} features")
    res = manager.predict(data.features)
    if not res: raise HTTPException(400, "Model not trained yet")
    return res