# Trabajo Práctico 2: Clasificación de Fashion-MNIST con PyTorch

## Estructura del Proyecto

```
rrnn2025-tp2/
├── bibliografia/              # Referencias bibliográficas y materiales de estudio
│   ├── README.md
│   ├── 01_pytorch_fundamentos.md
│   ├── 02_fashion_mnist.md
│   ├── 03_redes_feedforward.md
│   └── paper-fashion-MNIST.md
├── images/                    # Figuras y gráficos generados para el informe
│   ├── comparacion_*.png      # Gráficos de comparación de hiperparámetros
│   ├── curvas_entrenamiento_*.png
│   ├── matriz_confusion_*.png
│   └── fashion_mnist_ejemplos.png
├── informe_cientifico/        # Informe científico dividido en secciones
│   ├── README.md
│   ├── 01_titulo_autores.md
│   ├── 02_resumen.md
│   ├── 03_introduccion.md
│   ├── 04_teoria.md
│   ├── 05_resultados.md
│   ├── 06_discusion.md
│   └── 07_referencias.md
├── MNIST_data/                # Dataset Fashion-MNIST descargado (se genera automáticamente)
├── analisis_hiperparametros.md # Análisis completo de hiperparámetros y resultados
├── PLAN_DE_TRABAJO.md         # Plan detallado de trabajo
├── tp2_trabajo.ipynb          # Notebook principal de trabajo (EJECUTAR ESTE)
├── redes_neuronales_2025_guia_11.ipynb  # Notebook guía 11: Feedforward (referencia)
├── redes_neuronales_2025_guia_12.ipynb  # Notebook guía 12: Autoencoder (referencia)
├── requirements.txt           # Dependencias del proyecto
├── tp2-2025-1-md              # Enunciado del trabajo práctico
└── tp2-2025-1.pdf             # Enunciado del trabajo práctico (PDF)
```

## Descripción

Este trabajo práctico se enfoca en la implementación y entrenamiento de redes neuronales feedforward multicapa para clasificar imágenes del dataset Fashion-MNIST utilizando PyTorch.

### Objetivos principales:
- Implementar una red neuronal feedforward multicapa
- Entrenar el modelo para clasificar imágenes de Fashion-MNIST
- Analizar el efecto de diferentes hiperparámetros
- Evaluar el rendimiento mediante curvas de entrenamiento y validación

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- CUDA 12.1 (opcional, para aceleración GPU con NVIDIA)
- Git (para clonar el repositorio)

### Pasos de Instalación

1. **Crear un entorno virtual** (recomendado):
```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate  # En Windows
```

2. **Instalar dependencias con PyTorch y CUDA**:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

**Nota importante**: El flag `--extra-index-url` es necesario para instalar PyTorch con soporte CUDA. Si no tienes GPU NVIDIA o prefieres la versión CPU, puedes instalar PyTorch por separado:

```bash
# Para CPU solamente
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Luego instalar el resto
pip install -r requirements.txt
```

3. **Configurar el kernel de Jupyter** para usar el entorno virtual:
```bash
python -m ipykernel install --user --name=rrnn2025-tp2 --display-name "Python (rrnn2025-tp2)"
```

4. **Verificar la instalación**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## Uso

### Ejecutar el Notebook Principal

1. **Abrir Jupyter Notebook o JupyterLab**:
```bash
jupyter notebook
# o
jupyter lab
```

2. **Abrir el notebook principal**: `tp2_trabajo.ipynb`

3. **Seleccionar el kernel correcto**:
   - En Jupyter: `Kernel` → `Change Kernel` → `Python (rrnn2025-tp2)`
   - En VS Code/Cursor: Click en el selector de kernel (arriba a la derecha) → `Python (rrnn2025-tp2)`

4. **Ejecutar las celdas en orden**:
   - El notebook está estructurado en secciones numeradas
   - Cada sección corresponde a un ejercicio de la Guía 11
   - Las figuras se guardan automáticamente en `images/`

### Estructura del Notebook

El notebook `tp2_trabajo.ipynb` está organizado en las siguientes secciones:

- **Sección 1**: Importación de librerías
- **Sección 2**: Configuración del dispositivo (CPU/GPU)
- **Sección 3**: Carga y exploración del dataset Fashion-MNIST
- **Sección 4**: Definición de la arquitectura de la red neuronal
- **Sección 5**: Funciones de entrenamiento y validación
- **Sección 6**: Entrenamiento básico
- **Sección 7**: Visualización de resultados básicos
- **Sección 8**: Análisis de hiperparámetros (6 experimentos)
- **Sección 9**: Generación de figuras finales

### Archivos de Referencia

- **`redes_neuronales_2025_guia_11.ipynb`**: Guía de referencia para redes feedforward
- **`redes_neuronales_2025_guia_12.ipynb`**: Guía de autoencoders (no necesario para este TP)
- **`PLAN_DE_TRABAJO.md`**: Plan detallado paso a paso del trabajo
- **`analisis_hiperparametros.md`**: Análisis completo de todos los experimentos y resultados

### Generación de Figuras

Todas las figuras se guardan automáticamente en `images/` cuando ejecutas las celdas correspondientes:
- Curvas de entrenamiento
- Matrices de confusión
- Comparaciones de hiperparámetros
- Visualizaciones de imágenes reales vs predichas

## Resultados y Análisis

### Modelo Final

El modelo final entrenado con los mejores hiperparámetros alcanzó:
- **Precisión en validación**: 87.99%
- **Configuración óptima**:
  - Optimizador: ADAM
  - Learning Rate: 0.001
  - Batch Size: 32
  - Épocas: 30
  - Dropout: 0.2

### Análisis de Hiperparámetros

El archivo `analisis_hiperparametros.md` contiene:
- Explicación detallada de cada hiperparámetro
- Resultados de todos los experimentos realizados
- Comparaciones y análisis
- Resultados del modelo final
- Visualizaciones de todos los experimentos

### Experimentos Realizados

1. **Variación de Learning Rate**: 0.0001, 0.001, 0.01
2. **Comparación de Optimizadores**: SGD vs ADAM
3. **Variación de Dropout**: 0.0, 0.2, 0.4, 0.6
4. **Variación de Arquitectura**: (64,32), (128,64), (256,128), (512,256)
5. **Variación de Épocas**: 5, 10, 15, 20, 30
6. **Variación de Batch Size**: 32, 64, 100, 128, 256

## Informe Científico

El informe está dividido en secciones independientes en la carpeta `informe_cientifico/`. Ver el README en esa carpeta para instrucciones de compilación.

**Estructura del informe**:
- Título y autores
- Resumen
- Introducción
- Teoría
- Resultados
- Discusión
- Referencias

## Bibliografía

Los materiales de estudio y referencias están organizados en la carpeta `bibliografia/`. Cada archivo markdown contiene notas y resúmenes sobre diferentes temas relevantes:
- Fundamentos de PyTorch
- Dataset Fashion-MNIST
- Redes feedforward
- Paper original de Fashion-MNIST

## Solución de Problemas

### Error: "No module named 'matplotlib'"
**Solución**: Asegúrate de haber activado el entorno virtual y seleccionado el kernel correcto de Jupyter.

### Error: CUDA no disponible
**Solución**: Verifica que tengas una GPU NVIDIA y los drivers instalados. Si no tienes GPU, el código funcionará en CPU automáticamente.

### Kernel de Jupyter no aparece
**Solución**: Ejecuta nuevamente:
```bash
python -m ipykernel install --user --name=rrnn2025-tp2 --display-name "Python (rrnn2025-tp2)"
```

### Dataset no se descarga
**Solución**: El dataset se descarga automáticamente la primera vez que ejecutas el notebook. Asegúrate de tener conexión a internet.

## Notas Importantes

- **El informe no debe exceder 4 páginas**
- **Formato de entrega**: PDF
- **Nombre del archivo**: `grupo-número_apellido1_apellido2_apellido3_TP2_RN_2025.pdf`
- **Dataset**: Se descarga automáticamente en `MNIST_data/` la primera vez
- **Tiempo de entrenamiento**: Los experimentos pueden tardar varias horas dependiendo del hardware
- **GPU recomendada**: Para acelerar el entrenamiento, se recomienda usar GPU NVIDIA con CUDA

## Contacto y Contribuciones

Para dudas sobre el proyecto, consultar:
- El plan de trabajo: `PLAN_DE_TRABAJO.md`
- El análisis de hiperparámetros: `analisis_hiperparametros.md`
- Las guías de referencia: `redes_neuronales_2025_guia_11.ipynb`

