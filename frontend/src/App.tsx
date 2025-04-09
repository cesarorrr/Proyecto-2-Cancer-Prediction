import { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import './App.css'; // O elimina si usarás Tailwind completamente

interface FormData {
  edad: string;
  nivel_biomarcador_x: string;
  historial_familiar: boolean;
  resultado_analitica_y: string;
  // Agrega más campos aquí si es necesario
}

function App() {
  const [formData, setFormData] = useState<FormData>({
    edad: '',
    nivel_biomarcador_x: '',
    historial_familiar: false,
    resultado_analitica_y: '',
  });

  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setPredictionResult(null);

    const apiUrl =
      import.meta.env.VITE_API_URL ||
      process.env.REACT_APP_API_URL ||
      'http://localhost:8000';

    try {
      const dataToSend = {
        edad: parseInt(formData.edad, 10),
        nivel_biomarcador_x: parseFloat(formData.nivel_biomarcador_x),
        historial_familiar: formData.historial_familiar,
        resultado_analitica_y: parseFloat(formData.resultado_analitica_y),
        // Asegúrate de convertir más campos si los agregas
      };

      if (
        isNaN(dataToSend.edad) ||
        isNaN(dataToSend.nivel_biomarcador_x) ||
        isNaN(dataToSend.resultado_analitica_y)
      ) {
        throw new Error('Por favor, introduce valores numéricos válidos.');
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

  return (
    <div className="App">
      <h1>Predicción de Probabilidad de Cáncer</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="edad">Edad:</label>
          <input
            id="edad"
            type="number"
            name="edad"
            value={formData.edad}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <label htmlFor="nivel_biomarcador_x">Nivel Biomarcador X:</label>
          <input
            id="nivel_biomarcador_x"
            type="number"
            step="0.1"
            name="nivel_biomarcador_x"
            value={formData.nivel_biomarcador_x}
            onChange={handleChange}
            required
          />
        </div>
        <div className="checkbox-group">
          <label htmlFor="historial_familiar">Historial Familiar:</label>
          <input
            id="historial_familiar"
            type="checkbox"
            name="historial_familiar"
            checked={formData.historial_familiar}
            onChange={handleChange}
          />
        </div>
        <div>
          <label htmlFor="resultado_analitica_y">Resultado Analítica Y:</label>
          <input
            id="resultado_analitica_y"
            type="number"
            step="0.1"
            name="resultado_analitica_y"
            value={formData.resultado_analitica_y}
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
