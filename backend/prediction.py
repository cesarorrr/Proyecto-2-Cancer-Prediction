#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prediction.py — Inferencia unitaria con modelos de supervivencia de cáncer de colon
===================================================================================

Uso básico (PowerShell / CMD / Bash)
------------------------------------
# ► solo variables tabulares\python prediction.py --json data/patient_001.json

# ► variables tabulares + radiografía
python prediction.py --json data/patient_001.json --image data/imagenes_colon/colonca1.jpeg

(En PowerShell evita el salto de línea con "\`" o pon el comando en una sola línea).

El script resuelve dos problemas detectados al primer intento de ejecución:
  1. *PerformanceWarning* debido a la inserción columna-a-columna de 1280 embeddings.
  2. *ValueError* de XGBoost por desajuste de *feature names* entre el modelo híbrido entrenado y los datos de inferencia.

Solución implementada
----------------------
*   Se crea un *DataFrame* con **exactamente** las columnas que espera el modelo
    híbrido (`MODEL_HYB.get_booster().feature_names`). Las columnas ausentes se
    rellenan a 0, las sobrantes se descartan.
*   Los 1280 valores del embedding se añaden de una sola vez usando
    `pd.concat`, eliminando la fragmentación.

Salida de ejemplo
-----------------
```
▶ Probabilidad TABULAR : 0.23
▶ Probabilidad HÍBRIDO : 0.17
```

("*Valores meramente ilustrativos*").
"""
from __future__ import annotations
import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ────────────────────────── parámetros coherentes con train_model.py ───────────────
EFF_SHAPE = (224, 224)
EMB_SIZE  = 1280
TAB_VARS  = [
    "Age", "tumor_size", "relapse", "Family history",
    "inflammatory_bowel_disease", "cancer_stage", "obesity"
]

CAT_MAP = {
    "relapse":                    {"No": 0, "Yes": 1},
    "Family history":             {"No": 0, "Yes": 1},
    "inflammatory_bowel_disease": {"No": 0, "Yes": 1},
    "obesity":                    {"Normal": 0, "Overweight": 1, "Obese": 2},
    "cancer_stage":               {"I": 1, "II": 2, "III": 3, "IV": 4},
    "Sexo":                       {"F": 0, "M": 1},
    "smoke":                      {"No": 0, "Yes": 1},
    "alcohol":                    {"No": 0, "Yes": 1},
    "diet":                       {"Low": 0, "Moderate": 1, "High": 2},
    "Screening_History":          {"Never": 0, "Irregular": 1, "Regular": 2},
    "Healthcare_Access":          {"Low": 0, "Moderate": 1, "High": 2},
}

BASE_DIR = Path(__file__).resolve().parent
RES_DIR  = BASE_DIR / "results"

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────── carga modelos entrenados ───────────────────────────────
with (RES_DIR / "model_tabular.pkl").open("rb") as f:
    MODEL_TAB = pickle.load(f)

with (RES_DIR / "model_hybrid.pkl").open("rb") as f:
    MODEL_HYB = pickle.load(f)

# ────────────────────────── pre-procesado idéntico al entrenamiento ────────────────
def _apply_cat_map(df: pd.DataFrame) -> None:
    """Convierte categorías a numérico y asegura *dtype* float32 sin NaN"""
    for col, mapping in CAT_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # todo en float32 (coincide con XGB hist)
    df[:] = df.astype(np.float32)

# ────────────────────────── EfficientNetB0 para embeddings ─────────────────────────
def _load_effnet():
    from tensorflow.keras.applications import EfficientNetB0
    return EfficientNetB0(weights="imagenet", include_top=False,
                          pooling="avg", input_shape=(*EFF_SHAPE, 3))

def _pre(img: Image.Image):
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img_to_array(img.resize(EFF_SHAPE)))

def extract_embedding(img_path: str | None) -> np.ndarray:
    """1280-D embedding; vector de ceros si no se pasa imagen"""
    if img_path is None:
        return np.zeros(EMB_SIZE, dtype=np.float32)
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    effnet = _load_effnet()
    arr = _pre(Image.open(img_path).convert("RGB"))[None, ...]
    return effnet.predict(arr, verbose=0)[0].astype(np.float32)

# ────────────────────────── predicción unitaria ────────────────────────────────────
def _build_hybrid_dataframe(df_tab: pd.DataFrame, emb_vec: np.ndarray) -> pd.DataFrame:
    """Crea un DataFrame *exactamente* con las columnas que espera el modelo híbrido."""
    # columnas esperadas por el booster (en el mismo orden)
    try:
        expected_cols: list[str] = MODEL_HYB.get_booster().feature_names
    except AttributeError:  # Fallback scikit-learn >=1.2
        expected_cols = list(MODEL_HYB.feature_names_in_)

    # base toda a ceros (float32)
    df_hyb = pd.DataFrame(
        np.zeros((1, len(expected_cols)), dtype=np.float32),
        columns=expected_cols
    )

    # 1️⃣ variables tabulares que coinciden
    for col in df_tab.columns.intersection(df_hyb.columns):
        df_hyb.loc[0, col] = df_tab.at[0, col]

    # 2️⃣ embeddings (emb_0 … emb_1279)
    emb_cols = [c for c in expected_cols if c.startswith("emb_")]
    if emb_cols:
        df_hyb.loc[0, emb_cols] = emb_vec[: len(emb_cols)]

    return df_hyb

def predict_probabilities(json_path: str, image_path: str | None = None) -> dict[str, float]:
    """Devuelve un diccionario con las probabilidades de ambos modelos."""
    with open(json_path, "r", encoding="utf-8") as f:
        user_dict = json.load(f)

    df_tab = pd.DataFrame([user_dict])
    _apply_cat_map(df_tab)

    # asegurar columnas mínimas para el modelo tabular
    for col in TAB_VARS:
        if col not in df_tab.columns:
            df_tab[col] = 0

    prob_tab = float(MODEL_TAB.predict_proba(df_tab[TAB_VARS])[:, 1][0])

    # híbrido
    emb_vec = extract_embedding(image_path)
    df_hyb  = _build_hybrid_dataframe(df_tab, emb_vec)
    prob_hyb = float(MODEL_HYB.predict_proba(df_hyb)[:, 1][0])

    return {"tabular": prob_tab, "hybrid": prob_hyb}

# ────────────────────────── CLI ────────────────────────────────────────────────────

def _main_cli():
    ap = argparse.ArgumentParser(description="Predicción (un paciente)")
    ap.add_argument("--json", required=True, help="JSON con variables tabulares")
    ap.add_argument("--image", help="Ruta de radiografía (opcional)")
    args = ap.parse_args()

    probs = predict_probabilities(args.json, args.image)
    print(f"\n▶ Probabilidad TABULAR : {probs['tabular']:.3f}")
    print(f"▶ Probabilidad HÍBRIDO : {probs['hybrid']:.3f}")

if __name__ == "__main__":
    _main_cli()
