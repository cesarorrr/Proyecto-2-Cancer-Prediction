#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_model.py — Supervivencia cáncer de colon
=============================================
• Tabular optimizado (RandomForest / XGBoost con GridSearchCV)
• Híbrido (tabular + embeddings EfficientNetB0)
• Aumento ligero de positivos (flip / rot / brillo / contraste)
• Métricas, matriz de confusión e importancia de variables
• Artefactos en backend/results/
"""

from __future__ import annotations
import json, pickle, warnings
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────── parámetros globales ──────────────
RND          = 42
TARGET_COL   = "relapse"
EFF_SHAPE    = (224, 224)
EMB_SIZE     = 1280
AUG_PER_POS  = 6

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
IMG_DIR   = DATA_DIR / "imagenes_colon"
RES_DIR   = BASE_DIR / "results";  RES_DIR.mkdir(exist_ok=True)

# ────────────── variables + codificación fija ──────────────
TAB_VARS = ["Age", "tumor_size", "relapse", "Family history",
            "inflammatory_bowel_disease", "cancer_stage", "obesity"]

CAT_MAP  = {
    # objetivo + factores del paper
    "relapse":                   {"No": 0, "Yes": 1},
    "Family history":            {"No": 0, "Yes": 1},
    "inflammatory_bowel_disease":{"No": 0, "Yes": 1},
    "obesity":                   {"Normal": 0, "Overweight": 1, "Obese": 2},
    "cancer_stage":              {"I": 1, "II": 2, "III": 3, "IV": 4},
    # columnas extra del dataset (para el híbrido)
    "Sexo":                      {"F": 0, "M": 1},
    "smoke":                     {"No": 0, "Yes": 1},
    "alcohol":                   {"No": 0, "Yes": 1},
    "diet":                      {"Low": 0, "Moderate": 1, "High": 2},
    "Screening_History":         {"Never": 0, "Irregular": 1, "Regular": 2},
    "Healthcare_Access":         {"Low": 0, "Moderate": 1, "High": 2},
}

# ══════════════════ helpers CSV ══════════════════
def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, encoding="latin1")


def load_dataset() -> pd.DataFrame:
    h = _csv("historial_medico.csv")
    s = _csv("analisis_cancer.csv")
    i = _csv("historial_medico_imagenes.csv")

    for df in (h, s, i):
        df.rename(columns={c: "id" for c in ("ID", "Id", "id", "patient_id") if c in df.columns},
                  inplace=True)

    df = h.merge(s, on="id", how="left").merge(i[["id", "Imagename"]], on="id", how="left")
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' no existe en los CSV")
    df[TARGET_COL] = df[TARGET_COL].fillna("No")
    return df

# ══════════════════ EfficientNetB0 ══════════════════
def _load_effnet():
    from tensorflow.keras.applications import EfficientNetB0
    return EfficientNetB0(weights="imagenet", include_top=False, pooling="avg",
                          input_shape=(*EFF_SHAPE, 3))


def _img_path(name) -> Path | None:
    if pd.isna(name): return None
    base = str(name).strip(); digits = "".join(ch for ch in base if ch.isdigit())
    for cand in {base, f"colonca{digits}"}:
        for ext in (".jpeg", ".jpg", ".png"):
            p = IMG_DIR / f"{cand}{ext}"
            if p.exists(): return p
    return None


def _pre(img: Image.Image):
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img_to_array(img.resize(EFF_SHAPE)))


def extract_embeddings(df: pd.DataFrame, model) -> np.ndarray:
    embs = []
    for n in tqdm(df["Imagename"], desc="Embeddings", ncols=80):
        p = _img_path(n)
        if p is None:
            embs.append(np.zeros(EMB_SIZE, dtype=np.float32))
            continue
        arr = _pre(Image.open(p).convert("RGB"))[None, ...]
        embs.append(model.predict(arr, verbose=0)[0].astype(np.float32))
    return np.vstack(embs)

# ══════════════════ augment positivos ══════════════════
def _augment(img: Image.Image) -> Image.Image:
    img = img.rotate(np.random.uniform(-20, 20))
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))
    return img


def augment_dataset(df: pd.DataFrame, y: np.ndarray, effnet, emb0: np.ndarray) \
        -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feat = [c for c in df.columns if c not in ("id", "Imagename", TARGET_COL)]
    emap = dict(zip(df["id"], emb0))
    Xtab, Xemb, Y = [], [], []

    for _, row in df.iterrows():
        lab = int(str(row[TARGET_COL]).upper() == "YES"); pid = row["id"]
        Xtab.append(row[feat].to_dict()); Xemb.append(emap[pid]); Y.append(lab)

        if lab:
            p = _img_path(row["Imagename"])
            if not p: continue
            base = Image.open(p).convert("RGB")
            for _ in range(AUG_PER_POS):
                emb = effnet.predict(_pre(_augment(base))[None, ...], verbose=0)[0]
                Xtab.append(row[feat].to_dict()); Xemb.append(emb); Y.append(1)

    return pd.DataFrame(Xtab), np.vstack(Xemb), np.array(Y, int)

# ══════════════════ util común ══════════════════
def _apply_cat_map(df: pd.DataFrame) -> None:
    """Convierte todas las columnas categóricas a numérico de manera robusta."""
    for col, mapping in CAT_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # lo que quede como object → códigos
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # asegurar numérico y sin NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ══════════════════ TABULAR optimizado ══════════════════
def train_tabular(X_raw: pd.DataFrame, y: np.ndarray):
    _apply_cat_map(X_raw)
    X = X_raw[TAB_VARS].copy()

    cls_counts = Counter(y); spw = cls_counts[0] / max(1, cls_counts[1])

    search_space = {
        "RF": dict(
            model=RandomForestClassifier(random_state=RND, class_weight="balanced"),
            grid={
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2],
            },
        ),
        "XGB": dict(
            model=XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                random_state=RND, tree_method="hist",
                                scale_pos_weight=spw),
            grid={
                "n_estimators": [200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "min_child_weight": [1, 3],
            },
        ),
    }

    best_auc, best_est = 0, None
    for name, cfg in search_space.items():
        gs = GridSearchCV(cfg["model"], cfg["grid"], cv=5,
                          scoring="roc_auc", n_jobs=-1, verbose=0).fit(X, y)
        if gs.best_score_ > best_auc:
            best_auc, best_est = gs.best_score_, gs.best_estimator_
    print(f"▶ Tabular – mejor ROC-AUC CV: {best_auc:.3f}")
    best_est.fit(X, y)
    return best_est, X  # X procesado → importancia

# ══════════════════ HÍBRIDO ══════════════════
def train_hybrid(X_tab: pd.DataFrame, X_emb: np.ndarray, y: np.ndarray):
    _apply_cat_map(X_tab)  # ↓ ya todo numérico
    emb_cols = [f"emb_{i}" for i in range(X_emb.shape[1])]
    full = pd.concat([X_tab.reset_index(drop=True),
                      pd.DataFrame(X_emb, columns=emb_cols)], axis=1)

    spw = (y == 0).sum() / max(1, (y == 1).sum())
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          random_state=RND, tree_method="hist",
                          scale_pos_weight=spw,
                          n_estimators=600, learning_rate=0.05,
                          max_depth=5, subsample=0.8, colsample_bytree=0.8)
    model.fit(full, y)
    return model, full

# ══════════════════ evaluación ══════════════════
def _eval(name: str, model, X_raw: pd.DataFrame, y,
          fig: Path, is_hybrid: bool = False):
    _apply_cat_map(X_raw)
    X_eval = X_raw if is_hybrid else X_raw[TAB_VARS]

    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    met = {"accuracy": accuracy_score(y, y_pred),
           "precision": precision_score(y, y_pred, zero_division=0),
           "recall": recall_score(y, y_pred, zero_division=0),
           "f1": f1_score(y, y_pred, zero_division=0),
           "roc_auc": roc_auc_score(y, y_proba)}

    print(f"\n{name} – métricas")
    for k, v in met.items():
        print(f"   {k}: {v:.3f}")

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(colorbar=False)
    fig.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig, dpi=120, bbox_inches="tight")
    plt.close()
    return met


def save_imp(model, X_proc: pd.DataFrame, path: Path, top: int = 7):
    if hasattr(model, "feature_importances_"):
        imp = (pd.DataFrame({"feature": X_proc.columns,
                             "importance": model.feature_importances_})
               .sort_values("importance", ascending=False))
        imp.to_csv(path, index=False)
        (path.parent / "selected_features.txt").write_text(
            "\n".join(imp.head(top)["feature"])
        )

# ══════════════════ MAIN ══════════════════
def main():
    print("▶ Cargando datos…"); df = load_dataset()
    y = (df[TARGET_COL].str.upper() == "YES").astype(int).to_numpy()

    print("▶ Extrayendo embeddings EfficientNetB0…")
    eff = _load_effnet(); emb0 = extract_embeddings(df, eff)

    print("▶ Aumentando positivos…")
    X_tab_aug, X_emb_aug, y_aug = augment_dataset(df, y, eff, emb0)

    # -------- TABULAR --------
    Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(df.copy(), y, test_size=0.2,
                                                 stratify=y, random_state=RND)
    print("▶ Entrenando TABULAR…")
    mod_tab, X_proc_tr = train_tabular(Xtr_t[TAB_VARS].copy(), ytr_t)
    met_tab = _eval("Tabular", mod_tab, Xte_t[TAB_VARS].copy(), yte_t,
                    RES_DIR / "tabular_cm.png")
    save_imp(mod_tab, X_proc_tr, RES_DIR / "feat_imp_tab.csv")

    # -------- HÍBRIDO --------
    emb_cols = [f"emb_{i}" for i in range(X_emb_aug.shape[1])]
    full = pd.concat([X_tab_aug.reset_index(drop=True),
                      pd.DataFrame(X_emb_aug, columns=emb_cols)], axis=1)
    Xtr_h, Xte_h, ytr_h, yte_h = train_test_split(full, y_aug, test_size=0.2,
                                                 stratify=y_aug, random_state=RND)
    print("▶ Entrenando HÍBRIDO…")
    mod_hyb, _ = train_hybrid(Xtr_h.drop(columns=emb_cols),
                              Xtr_h[emb_cols].to_numpy(), ytr_h)
    met_hyb = _eval("Híbrido", mod_hyb, Xte_h[full.columns], yte_h,
                    RES_DIR / "hybrid_cm.png", is_hybrid=True)

    # -------- artefactos --------
    with open(RES_DIR / "model_tabular.pkl", "wb") as f:
        pickle.dump(mod_tab, f)
    with open(RES_DIR / "model_hybrid.pkl", "wb") as f:
        pickle.dump(mod_hyb, f)
    with open(RES_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"tabular": met_tab, "hybrid": met_hyb}, f, indent=4)

    print("✅ COMPLETADO – revisa backend/results/")


if __name__ == "__main__":
    main()
