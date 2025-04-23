import { useState, ChangeEvent, FormEvent, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Swal from 'sweetalert2';

interface FormData {
  Age: string;
  Hemoglobina: string;
  Plaquetas: string;
  cancer_stage: string;
  Tumor_Stage_Interaction: string;
}

function App() {
  const [formData, setFormData] = useState<FormData>({
    Age: '',
    Hemoglobina: '',
    Plaquetas: '',
    cancer_stage: '',
    Tumor_Stage_Interaction: '',
  });

  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setPredictionResult(null);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      const dataToSend = {
        features: {
          Age: parseInt(formData.Age, 10),
          Hemoglobina: parseFloat(formData.Hemoglobina),
          Plaquetas: parseFloat(formData.Plaquetas),
          cancer_stage: parseInt(formData.cancer_stage, 10),
          Tumor_Stage_Interaction: parseFloat(formData.Tumor_Stage_Interaction),
        },
      };

      if (
        Object.values(dataToSend.features).some((val) => isNaN(val as number))
      ) {
        throw new Error(
          'Por favor, introduce todos los valores correctamente.'
        );
      }

      const response = await axios.post(`${apiUrl}/predict`, dataToSend);
      setPredictionResult(response.data.probabilidad_cancer);
    } catch (err: any) {
      let errorMsg = 'Ocurrió un error al contactar el servidor.';
      if (axios.isAxiosError(err)) {
        if (err.response) {
          errorMsg = `Error del servidor (${err.response.status}): ${
            err.response.data?.detail || 'No se pudo procesar la solicitud.'
          }`;
        } else if (err.request) {
          errorMsg =
            'No se recibió respuesta del servidor. ¿Está funcionando y accesible?';
        } else {
          errorMsg = `Error en la solicitud: ${err.message}`;
        }
      } else {
        errorMsg = `Error inesperado: ${err.message}`;
      }
      setError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const checkServer = async () => {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      const res = await axios.get(`${apiUrl}/health`);
      if (res.status === 200) {
        Swal.close();
      } else {
        setTimeout(() => checkServer(), 5000);
      }
    } catch {
      setTimeout(() => checkServer(), 5000);
    }
  };

  useEffect(() => {
    Swal.fire({
      title: 'Verificando conexión con el servidor...',
      text: 'Por favor espere...',
      icon: 'info',
      allowOutsideClick: false,
      showConfirmButton: false,
      willOpen: () => {
        checkServer();
      },
      didOpen: () => {
        Swal.showLoading();
      },
    });
  }, []);

  return (
    <div className="App">
      <h1>Predicción de Probabilidad de Cáncer</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="Age">Edad:</label>
          <input
            id="Age"
            name="Age"
            type="number"
            value={formData.Age}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <label htmlFor="Hemoglobina">Hemoglobina:</label>
          <input
            id="Hemoglobina"
            name="Hemoglobina"
            type="number"
            step="0.1"
            value={formData.Hemoglobina}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <label htmlFor="Plaquetas">Plaquetas:</label>
          <input
            id="Plaquetas"
            name="Plaquetas"
            type="number"
            step="0.1"
            value={formData.Plaquetas}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <label htmlFor="cancer_stage">Etapa del Cáncer:</label>
          <input
            id="cancer_stage"
            name="cancer_stage"
            type="number"
            value={formData.cancer_stage}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <label htmlFor="Tumor_Stage_Interaction">
            Interacción Tumor/Etapa:
          </label>
          <input
            id="Tumor_Stage_Interaction"
            name="Tumor_Stage_Interaction"
            type="number"
            step="0.1"
            value={formData.Tumor_Stage_Interaction}
            onChange={handleChange}
            required
          />
        </div>

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Calculando...' : 'Obtener Predicción'}
        </button>
      </form>

      {isLoading && <p className="loading-message">Procesando...</p>}
      {error && <p className="error-message">Error: {error}</p>}
      {predictionResult !== null && (
        <div className="result">
          <h2>Resultado de la Predicción:</h2>
          <p>
            Probabilidad de Cáncer:{' '}
            <span>{(predictionResult * 100).toFixed(2)}%</span>
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
