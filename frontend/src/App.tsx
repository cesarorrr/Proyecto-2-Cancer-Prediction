import { useState, ChangeEvent, FormEvent, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Swal from 'sweetalert2';

interface FormDataFields {
  Age: string;
  cancer_stage: string;
  tumor_size: string;
  family_history: boolean; // ✅ era string
  inflammatory_bowel_disease: boolean; // ✅ era string
  obesity: string;
  photo?: File | null;
}

function App() {
  const [formData, setFormData] = useState<FormDataFields>({
    Age: '',
    cancer_stage: '',
    tumor_size: '',
    family_history: false, // ✅ antes era 'No'
    inflammatory_bowel_disease: false, // ✅ antes era 'No'
    obesity: 'Normal',
    photo: null,
  });

  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, files } = e.target as HTMLInputElement;

    if (name === 'photo' && files) {
      setFormData((prev) => ({
        ...prev,
        photo: files[0],
      }));
    } else {
      let parsedValue: any = value;

      // Convertir a booleano si es un campo booleano
      if (name === 'family_history' || name === 'inflammatory_bowel_disease') {
        parsedValue = value === 'Yes';
      }

      setFormData((prev) => ({
        ...prev,
        [name]: parsedValue,
      }));
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setPredictionResult(null);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      let image_base64: string | null = null;

      // ⚡ Convertir imagen a Base64 si se subió una
      if (formData.photo) {
        const fileReader = new FileReader();
        image_base64 = await new Promise((resolve, reject) => {
          fileReader.onload = () =>
            resolve(fileReader.result?.toString().split(',')[1] || ''); // Obtén la parte Base64
          fileReader.onerror = (error) => reject(error);
          fileReader.readAsDataURL(formData.photo!); // <- Aquí el "!" asegura a TS que no es null
        });
      }

      // Prepara el objeto JSON con los datos y la imagen en base64
      const jsonToSend = {
        features: {
          Age: Number(formData.Age),
          cancer_stage: Number(formData.cancer_stage),
          tumor_size: Number(formData.tumor_size),
          family_history: formData.family_history,
          inflammatory_bowel_disease: formData.inflammatory_bowel_disease,
          obesity: Number(formData.obesity), // ← conversión explícita
        },
        image_base64: image_base64 || null,
      };

      // Enviar los datos como JSON al servidor
      const response = await axios.post(`${apiUrl}/predict`, jsonToSend, {
        headers: {
          'Content-Type': 'application/json', // Cambiar a 'application/json'
        },
      });

      setPredictionResult(response.data.probabilidad_cancer);
    } catch (err: any) {
      let errorMsg = 'Ocurrió un error al contactar el servidor.';
      if (axios.isAxiosError(err)) {
        if (err.response) {
          errorMsg = `Error del servidor (${err.response.status}): ${
            err.response.data?.detail || 'No se pudo procesar la solicitud.'
          }`;
        } else if (err.request) {
          errorMsg = 'No se recibió respuesta del servidor.';
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

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      // También si necesitas enviar el archivo en el formData más adelante, puedes guardarlo.
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
      <h1>Predicción de Cáncer</h1>
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
          <label htmlFor="cancer_stage">Etapa del Cáncer:</label>
          <select
            id="cancer_stage"
            name="cancer_stage"
            value={formData.cancer_stage}
            onChange={handleChange}
            required>
            <option value="">Seleccione...</option>
            <option value="1">I</option>
            <option value="2">II</option>
            <option value="3">III</option>
          </select>
        </div>
        <div>
          <label htmlFor="tumor_size">Tamaño del Tumor (cm):</label>
          <input
            id="tumor_size"
            name="tumor_size"
            type="number"
            step="0.1"
            min="0"
            value={formData.tumor_size}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <label htmlFor="family_history">Antecedentes Familiares:</label>
          <select
            id="family_history"
            name="family_history"
            value={formData.family_history ? 'Yes' : 'No'}
            onChange={handleChange}
            required>
            <option value="Yes">Sí</option>
            <option value="No">No</option>
          </select>
        </div>
        <div>
          <label htmlFor="inflammatory_bowel_disease">
            Enfermedad Inflamatoria Intestinal:
          </label>
          <select
            id="inflammatory_bowel_disease"
            name="inflammatory_bowel_disease"
            value={formData.inflammatory_bowel_disease ? 'Yes' : 'No'}
            onChange={handleChange}
            required>
            <option value="Yes">Sí</option>
            <option value="No">No</option>
          </select>
        </div>
        <div>
          <label htmlFor="obesity">Obesidad:</label>
          <select
            id="obesity"
            name="obesity"
            value={formData.obesity}
            onChange={handleChange}
            required>
            <option value={0}>Normal</option>
            <option value={1}>Sobrepeso</option>
            <option value={2}>Obeso</option>
          </select>
        </div>
        <div>
          <label htmlFor="photo">Foto de Colonoscopía:</label>
          <input
            id="photo"
            name="photo"
            type="file"
            accept="image/jpeg,image/jpg"
            onChange={handleFileChange}
          />
        </div>
        {selectedImage && (
          <div className="image-preview">
            <img
              src={selectedImage}
              alt="Vista previa"
              className="preview-img"
            />
          </div>
        )}
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Enviando...' : 'Obtener Predicción'}
        </button>
      </form>

      {isLoading && <p className="loading-message">Procesando...</p>}
      {error && <p className="error-message">Error: {error}</p>}
      {predictionResult !== null && (
        <div className="result">
          <h2>Resultado:</h2>
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
