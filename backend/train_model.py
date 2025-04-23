# quick_train.py
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

# ── 1. Definir rutas ──────────────────────────────────────────────────────────
DATA_DIR    = pathlib.Path(__file__).parent / "data"
MODEL_PATH  = pathlib.Path(__file__).parent / "modelo_cancer_tabular.pkl"
TARGET_COL  = "relapse"
TOP_K       = 5  # número de variables a retener

print(f"Buscando datos en: {DATA_DIR}")
print(f"Se guardará el modelo en: {MODEL_PATH}")

# ── 2. Carga de datos ─────────────────────────────────────────────────────────
# Historial médico
print("Cargando historial_medico.csv...")
hist = pd.read_csv(DATA_DIR / "historial_medico.csv", encoding="utf-8")
if "Id" in hist.columns:
    hist.rename(columns={"Id": "id"}, inplace=True)
print(f"  → {len(hist)} registros")

# Análisis de sangre (formato con ';')
print("Cargando analisis_sangre_dataset.csv...")
sangre = pd.read_csv(
    DATA_DIR / "analisis_sangre_dataset.csv",
    encoding="cp1252", sep=";", decimal="."
)
# si Pandas no separó bien las columnas:
if len(sangre.columns) == 1 and ";" in sangre.iloc[0,0]:
    cols = sangre.columns[0].split(";")
    rows = [r[0].split(";") for _, r in sangre.iterrows()]
    sangre = pd.DataFrame(rows, columns=cols)
# renombrar y convertir id
sangre.rename(columns={sangre.columns[0]: "id"}, inplace=True)
sangre["id"] = pd.to_numeric(sangre["id"], errors="coerce").astype(int)
print(f"  → {len(sangre)} registros")

# Análisis de cáncer
print("Cargando analisis_cancer.csv...")
cancer = pd.read_csv(DATA_DIR / "analisis_cancer.csv", encoding="utf-8")
print(f"  → {len(cancer)} registros")

# ── 3. Unión y preparación ───────────────────────────────────────────────────
print("Uniendo datasets...")
df = (
    hist.merge(sangre,  on="id", how="inner")
        .merge(cancer, on="id", how="inner")
)
print(f"  → dataset: {df.shape[0]} filas × {df.shape[1]} columnas")

# target binaria
df[TARGET_COL] = (
    df[TARGET_COL].astype(str)
                .str.strip()
                .str.lower()
                .eq("yes")
).astype(int)
print("Distribución de clases:", df[TARGET_COL].value_counts().to_dict())

# separar X / y
X = df.drop(columns=["id", TARGET_COL])
y = df[TARGET_COL]

# ── 4. Train-test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train/test: {X_train.shape[0]}/{X_test.shape[0]} registros")

# ── 5. Preprocesamiento y pipeline ───────────────────────────────────────────
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(f"  • categóricas: {len(cat_cols)}, numéricas: {len(num_cols)}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42
    ))
])

# ── 6. Entrenamiento ─────────────────────────────────────────────────────────
print("Entrenando modelo...")
pipeline.fit(X_train, y_train)

# ── 7. Umbral óptimo ─────────────────────────────────────────────────────────
print("Buscando umbral óptimo para precisión ≥ 0.75...")
y_proba = pipeline.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
idx75 = np.where(precision[:-1] >= 0.75)[0]
if len(idx75):
    # entre ellos, escoger el que maximiza recall
    best = idx75[np.argmax(recall[idx75])]
else:
    # si ninguno, escoger umbral de máxima precisión
    best = np.argmax(precision[:-1])
threshold = float(thresholds[best])
print(f"Umbral óptimo: {threshold:.4f} (precisión={precision[best]:.3f}, recall={recall[best]:.3f})")

# ── 8. Selección de top-K features ───────────────────────────────────────────
print("Calculando importancias de características...")
# obtener nombres de las columnas tras el preprocesador
feat_names = pipeline \
    .named_steps["preprocessor"] \
    .get_feature_names_out()
importances = pipeline \
    .named_steps["classifier"] \
    .feature_importances_
fi_df = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
      .sort_values("importance", ascending=False)
)
top_features = fi_df["feature"].head(TOP_K).tolist()
print(f"Top {TOP_K} features:", top_features)

# ── 9. Guardar modelo + artefactos ───────────────────────────────────────────
artifact = {
    "model": pipeline,
    "feature_names": top_features,
    "threshold": threshold
}
joblib.dump(artifact, MODEL_PATH, compress=3)
print(f"✔ Modelo guardado en {MODEL_PATH}")
# también para compatibilidad multimodal
MULTI = pathlib.Path(__file__).parent / "modelo_cancer_multimodal.pkl"
joblib.dump(artifact, MULTI, compress=3)
print(f"✔ Compatible multimodal en {MULTI}\n")

print("¡Entrenamiento completado con éxito!")
print("Ahora, al llamar al endpoint `/predict`, envía solo estas variables (y la imagen opcional).")
