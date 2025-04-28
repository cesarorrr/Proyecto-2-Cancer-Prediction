# main.py
import os
import pathlib
import joblib
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from typing import Optional, Type, Dict, Any
import base64
from io import BytesIO
from PIL import Image
import uvicorn

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cancer_prediction_api")

# Configuración de la app
app = FastAPI(title="Cancer Prediction API", version="1.0.0")
origins = [
    "https://proyecto-2-cancer-prediction-1.onrender.com",
    "http://localhost:3000",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas a modelos
HERE = pathlib.Path(__file__).parent.resolve()
TABULAR_MODEL_PATH = HERE / "modelo_cancer_tabular.pkl"
MULTIMODAL_MODEL_PATH = HERE / "modelo_cancer_multimodal.pkl"

# Variables globales para los modelos
tabular_model = None
tabular_features = []
tabular_threshold = 0.5
multimodal_model = None
multimodal_features = []
multimodal_threshold = 0.5

# Cargar los modelos
@app.on_event("startup")
async def load_models():
    logger.info("🔄 Intentando cargar modelos...")
    global tabular_model, tabular_features, tabular_threshold
    global multimodal_model, multimodal_features, multimodal_threshold
    print(f"¿Existe TABULAR_MODEL_PATH? {TABULAR_MODEL_PATH.exists()}")
    print(f"Ruta absoluta: {TABULAR_MODEL_PATH.absolute()}")
    print(f"¿Existe MULTIMODAL_MODEL_PATH? {MULTIMODAL_MODEL_PATH.exists()}")
    print(f"Ruta absoluta: {MULTIMODAL_MODEL_PATH.absolute()}")
    # Cargar modelo tabular
    if TABULAR_MODEL_PATH.exists():
        try:
            artifact = joblib.load(TABULAR_MODEL_PATH)
            tabular_model = artifact.get("model")
            tabular_features = artifact.get("feature_names", [])
            tabular_threshold = artifact.get("threshold", 0.5)
            logger.info(f"Modelo tabular cargado con {len(tabular_features)} features y umbral {tabular_threshold:.4f}")
        except Exception as e:
            logger.error(f"Error al cargar modelo tabular: {e}")
    else:
        logger.warning(f"Modelo tabular no encontrado en {TABULAR_MODEL_PATH}")
        logger.warning("Ejecuta 'python quick_train.py' para entrenar el modelo primero")
    
    # Cargar modelo multimodal si está disponible
    if MULTIMODAL_MODEL_PATH.exists():
        try:
            multimodal_artifact = joblib.load(MULTIMODAL_MODEL_PATH)
            multimodal_model = multimodal_artifact.get("model")
            multimodal_features = multimodal_artifact.get("feature_names", [])
            multimodal_threshold = multimodal_artifact.get("threshold", 0.5)
            logger.info(f"Modelo multimodal cargado con {len(multimodal_features)} features y umbral {multimodal_threshold:.4f}")
        except Exception as e:
            logger.error(f"Error al cargar modelo multimodal: {e}")
    else:
        logger.info(f"Modelo multimodal no encontrado en {MULTIMODAL_MODEL_PATH}")

# Definir modelos de entrada/salida
class PredictionInput(BaseModel):
    features: Dict[str, Any] = Field(..., description="Datos clínicos como pares clave-valor")
    image_base64: Optional[str] = Field(None, description="Imagen opcional en formato base64")

class PredictionOutput(BaseModel):
    probabilidad_cancer: float = Field(..., example=0.85)
    prediccion: int = Field(..., example=1)
    umbral: float = Field(..., example=0.75)
    modelo_usado: str = Field(..., example="tabular")

@app.get("/")
def read_root():
    return {
        "message": "API de Predicción de Recaída de Cáncer", 
        "tabular_model": tabular_model is not None,
        "multimodal_model": multimodal_model is not None,
        "tabular_features": tabular_features
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models": {
            "tabular": {
                "loaded": tabular_model is not None,
                "features": len(tabular_features),
                "threshold": tabular_threshold
            },
            "multimodal": {
                "loaded": multimodal_model is not None,
                "features": len(multimodal_features) if multimodal_model else 0,
                "threshold": multimodal_threshold
            }
        }
    }

@app.post("/predict", response_model=PredictionOutput)
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    logger.info("📥 Datos recibidos:")
    logger.info(input_data)

    if tabular_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelos no cargados aún. Por favor ejecuta 'python quick_train.py' primero."
        )

    try:
        # Extraer datos
        features_dict = input_data.features
        image_base64 = input_data.image_base64
        
        # Mostrar en consola cada campo recibido
        logger.info("📄 Campos clínicos recibidos:")
        for key, value in features_dict.items():
            logger.info(f" - {key}: {value}")

        if image_base64:
            logger.info("🖼 Imagen recibida (base64): OK")
        else:
            logger.info("🖼 Imagen no enviada")

        # Preprocesar los datos para que sean numéricos o categóricos codificados
        df = pd.DataFrame([{
            "Age": int(features_dict.get("Age", 0)),
            "cancer_stage": int(features_dict.get("cancer_stage", 0)),
            "tumor_size": float(features_dict.get("tumor_size", 0.0)),
            "Family history": 1 if features_dict.get("Family history", "No") == "Yes" else 0,
            "inflammatory_bowel_disease": 1 if features_dict.get("inflammatory_bowel_disease", "No") == "Yes" else 0,
            "obesity": {"Normal": 0, "Overweight": 1, "Obese": 2}.get(features_dict.get("obesity", "Normal"), 0),
        }])

        logger.info(f"🧮 DataFrame procesado para predicción:\n{df}")

        # --- El resto sigue como ya lo tienes ---
        
        use_multimodal = (
            multimodal_model is not None 
            and image_base64 is not None
        )

        if use_multimodal:
            logger.info("Usando modelo multimodal para predicción")
            try:
                # Procesar imagen simulada
                image_features = {}
                for i in range(10):
                    image_features[f"img_feat_{i+1}"] = np.random.normal(0, 1)

                combined_df = pd.concat([
                    df.reset_index(drop=True),
                    pd.DataFrame([image_features]).reset_index(drop=True)
                ], axis=1)

                for col in multimodal_features:
                    if col not in combined_df.columns:
                        combined_df[col] = 0.0

                combined_df = combined_df[multimodal_features]

                proba = multimodal_model.predict_proba(combined_df)[0, 1]
                prediction = int(proba >= multimodal_threshold)

                return PredictionOutput(
                    probabilidad_cancer=float(proba),
                    prediccion=prediction,
                    umbral=float(multimodal_threshold),
                    modelo_usado="multimodal"
                )

            except Exception as e:
                logger.error(f"Error al usar modelo multimodal: {e}")
                logger.info("Usando modelo tabular como respaldo")
        
        logger.info("Usando modelo tabular para predicción")

        for col in tabular_features:
            if col not in df.columns:
                df[col] = 0.0

        df = df[tabular_features]

        proba = tabular_model.predict_proba(df)[0, 1]
        prediction = int(proba >= tabular_threshold)

        return PredictionOutput(
            probabilidad_cancer=float(proba),
            prediccion=prediction,
            umbral=float(tabular_threshold),
            modelo_usado="tabular"
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)