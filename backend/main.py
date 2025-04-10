import os
import joblib # Or import pickle if you used pickle
import pandas as pd # Or numpy, depending on your model's expected input
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Field can be used for examples in newer Pydantic/FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # Optional for local execution trigger

# --- App Configuration ---
app = FastAPI(title="Cancer Prediction API", version="1.0.0")

# --- CORS Configuration ---
# Allows your frontend (running on a different domain/port)
# to communicate with this API. Adjust origins for production!
origins = [
    "https://proyecto-2-cancer-prediction-1.onrender.com",
    "http://localhost:3000",  # Default React dev port
    "http://localhost:5173", # Default Vite dev port
    # Add your deployed frontend Render URL here later
    # "https://your-frontend-app-name.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Model Loading ---
MODEL_PATH = 'modelo_cancer.pkl'
model = None

@app.on_event("startup")
def load_model_on_startup():
    """Loads the model when the application starts."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        # Decide if the app should fail to start if the model isn't found
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    try:
        # Ensure you use joblib if saved with joblib, or pickle if used pickle
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # You might want the app to fail if the model cannot be loaded
        raise RuntimeError(f"Could not load model: {e}")

# --- Input Data Model Definition (Pydantic) ---
# IMPORTANT: Adjust these fields EXACTLY to the features
# your model expects, with the same names and data types.
class InputFeatures(BaseModel):
    edad: int
    nivel_biomarcador_x: float
    historial_familiar: bool # Example: 0 or 1, True or False
    resultado_analitica_y: float
    # ... add ALL features your model needs here

    # Example providing sample data for documentation (FastAPI/Pydantic >= 0.100/2.x style)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "edad": 55,
                    "nivel_biomarcador_x": 12.3,
                    "historial_familiar": True,
                    "resultado_analitica_y": 45.6,
                    # ... example values for all features
                }
            ]
        }
    }
    # Older Pydantic v1 style:
    # class Config:
    #     schema_extra = { ... }


# --- Output Data Model Definition (Pydantic) ---
class PredictionOutput(BaseModel):
    probabilidad_cancer: float = Field(..., example=0.85)


# --- Root Endpoint (Optional) ---
@app.get("/")
def read_root():
    """Returns a welcome message."""
    return {"message": "Welcome to the Cancer Prediction API"}

@app.get("/health")
def health_check():
    """Endpoint para verificar que el API está en funcionamiento."""
    return {"status": "ok"}

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionOutput)
async def predict_cancer(features: InputFeatures):
    """
    Recibe los datos del paciente y devuelve la probabilidad de cáncer.
    """
    global model

    if model is None:
        try:
            loaded_model = joblib.load("modelo_cancer.pkl")
            print("Modelo cargado:", type(loaded_model))
            # Si el modelo está dentro de un dict, extráelo
            if isinstance(loaded_model, dict):
                model = loaded_model.get("model") or loaded_model.get("pipeline")
            else:
                model = loaded_model
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"No se pudo cargar el modelo: {e}")

    if not hasattr(model, "predict_proba"):
        # raise HTTPException(status_code=501, detail="El modelo cargado no soporta 'predict_proba'.")
        raise HTTPException(status_code=200, detail="El modelo cargado no soporta 'predict_proba'.")

    try:
        # Convertir los datos de entrada a DataFrame
        data_dict = features.model_dump()  # Usa .dict() si estás en Pydantic v1
        data_df = pd.DataFrame([data_dict])

        # Realizar la predicción
        probabilities = model.predict_proba(data_df)
        cancer_probability = probabilities[0][1]

        # Retornar la probabilidad
        return PredictionOutput(probabilidad_cancer=cancer_probability)

    except KeyError as e:
        print(f"Key error en los datos: {e}")
        raise HTTPException(status_code=422, detail=f"Campo de entrada inválido o faltante: {e}")
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción: {e}")


# --- Optional: Entry point for local execution ---
# Render will use Gunicorn, not this directly.
if __name__ == "__main__":
   print("Starting Uvicorn server on http://127.0.0.1:8000")
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)