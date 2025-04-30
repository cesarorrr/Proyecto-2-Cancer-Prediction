# 🧐 Sistema de Predicción de Cáncer de Colon

Bienvenido al repositorio del **Sistema Inteligente de Predicción de Recaída por Cáncer de Colon**, un proyecto desarrollado por el **Grupo ACO (Abel Pérez, César Rodríguez y Oriol Fernández )**. Esta solución integra técnicas avanzadas de Machine Learning y Deep Learning para brindar una evaluación médica precisa basada en datos clínicos y, opcionalmente, imágenes médicas.

---

## 🚀 Demo Online

- **Frontend (React + Vite):**  🔗 [https://proyecto-2-cancer-prediction-1.onrender.com](https://proyecto-2-cancer-prediction-1.onrender.com)

- **Backend (FastAPI):**  🔗 [https://proyecto-2-cancer-prediction.onrender.com](https://proyecto-2-cancer-prediction.onrender.com)

---

## 🧬 ¿Cómo funciona?

🔹 Los usuarios ingresan datos clínicos (edad, etapa del cáncer, obesidad, etc.)🔹 Opcionalmente, pueden subir una imagen de colonoscopía🔹 El sistema predice la probabilidad de recaída usando:

- Un modelo **tabular** si no hay imagen
- Un modelo **híbrido (tabular + imagen)** si se sube imagen válida

La respuesta incluye la probabilidad de recaída, tipo de modelo usado y umbral aplicado.

---

## 🖼️ Interfaz Web

📋 **Formulario de entrada:**

- Edad
- Etapa del cáncer
- Tamaño del tumor
- Historia familiar
- Enfermedad inflamatoria
- Nivel de obesidad
- Imagen de colonoscopía

📸 **Vista previa automática de imagen subida**📊 **Resultado mostrado como probabilidad (%) de recaída**

---

## 🧪 Tecnologías Usadas

### Frontend

- ⚛️ React (TypeScript)
- ⚡ Vite
- 🥃 SweetAlert2 (feedback al usuario)
- 🧪 Axios (conexión con backend)

### Backend

- 🐍 Python 3.10
- 🚀 FastAPI
- 📆 joblib, sklearn, XGBoost
- 🧠 EfficientNetB0 para embeddings de imagen
- 🖼️ PIL, base64 para procesamiento de imágenes

---

## 📁 Estructura del Proyecto

```
📆 cancer-prediction-project
🗁️ frontend/
├── src/App.tsx
└── .env (VITE_API_URL)
🗁️ backend/
├── main.py
├── models/
│   ├── model_tabular.pkl
│   └── model_hybrid.pkl
└── results/
    ├── metrics.json
    └── confusion_matrices.png
```

---

## ⚒️ Configuración Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/cancer-prediction.git
cd cancer-prediction
```

### 2. Iniciar el backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Iniciar el frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Variables necesarias

En el archivo `.env` del frontend:

```
VITE_API_URL=http://localhost:8000
```

---

## 📊 Modelos Incluidos

| Modelo  | Entrada        | Precisión esperada |
| ------- | -------------- | ------------------ |
| Tabular | Datos clínicos | >85% AUC-ROC       |
| Híbrido | Datos + Imagen | >90% AUC-ROC       |

🧠 Los modelos han sido entrenados usando GridSearchCV, embedding con EfficientNetB0, y técnicas de aumento de datos para imágenes reales.

---

## 🧲 Uso vía línea de comandos

```bash
python prediction.py --json paciente_001.json --image colonoscopia.jpg
```

---

## 🐞 Errores Comunes

| Error                  | Causa                | Solución                        |
| ---------------------- | -------------------- | ------------------------------- |
| No se recibe respuesta | Backend no activo    | Verifica URL o inicia `main.py` |
| Imagen inválida        | Formato no soportado | Usa `.jpg` o `.jpeg`            |
| Modelo no cargado      | Error en FastAPI     | Esperar 5s o reiniciar app      |

---

## 🧠 Créditos

Proyecto desarrollado por **Grupo ACO**📍 Abel Pérez · César Rodríguez · Oriol Fernández

Contratado por entidades médicas para apoyar el diagnóstico temprano del cáncer de colon.

---

## 📃 Licencia

MIT License – Este software es libre de uso, distribución y modificación.*Por favor, incluye atribución a los autores en caso de uso.*
