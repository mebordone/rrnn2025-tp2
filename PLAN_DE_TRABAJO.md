# Plan de Trabajo Detallado - TP2: Clasificación de Fashion-MNIST con PyTorch

## Información del Entorno
- **Sistema**: Linux Mint 22.1 x86_64
- **GPU**: NVIDIA GeForce RTX 3060 Lite Hash Rate
- **CPU**: AMD Ryzen 9 5900X
- **Trabajo**: Local (no Colab)
- **Formato informe**: Markdown → PDF con pandoc

---

## FASE 0: Instalación y Configuración Inicial (60-90 min)

### Paso 0.1: Instalar PyTorch con soporte CUDA

**Objetivo**: Instalar PyTorch con soporte para GPU NVIDIA RTX 3060

**Pasos**:
1. Verificar versión de CUDA instalada:
   ```bash
   nvidia-smi
   ```
   - Anotar la versión de CUDA mostrada (probablemente 11.x o 12.x)

2. Visitar https://pytorch.org/get-started/locally/
   - Seleccionar: Linux, Pip, Python, CUDA (versión detectada)
   - Copiar el comando de instalación sugerido

3. Activar entorno virtual (si existe) o crear uno nuevo:
   ```bash
   cd /home/mebordone/Documentos/Estudios/Doctorado/2025-RRNN/rrnn2025-tp/rrnn25-tp2/rrnn2025-tp2
   python -m venv .venv  # Si no existe
   source .venv/bin/activate
   ```

4. Instalar PyTorch con CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Ajustar según la versión de CUDA detectada)

5. Verificar instalación:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

**Resultado esperado**: PyTorch instalado y detectando la GPU RTX 3060

---

### Paso 0.2: Instalar dependencias restantes

**Objetivo**: Instalar todas las librerías necesarias

**Pasos**:
1. Instalar desde requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

2. Instalar librerías adicionales si faltan:
   ```bash
   pip install scikit-learn seaborn  # Para matriz de confusión y visualizaciones
   ```

3. Verificar instalación de Jupyter:
   ```bash
   pip install jupyter notebook
   ```

**Resultado esperado**: Todas las dependencias instaladas

---

### Paso 0.3: Crear notebook de trabajo estructurado

**Objetivo**: Crear un notebook organizado por secciones para ejecutar celdas individualmente

**Estructura sugerida del notebook**:
- **Sección 1**: Importación de librerías
- **Sección 2**: Configuración del dispositivo (CPU/GPU)
- **Sección 3**: Carga y exploración del dataset
- **Sección 4**: Definición de la arquitectura de la red
- **Sección 5**: Funciones de entrenamiento y validación
- **Sección 6**: Entrenamiento básico
- **Sección 7**: Visualización de resultados básicos
- **Sección 8**: Análisis de hiperparámetros
- **Sección 9**: Generación de figuras finales

**Pasos**:
1. Crear nuevo notebook: `tp2_trabajo.ipynb`
2. Organizar en celdas markdown (títulos de sección) y celdas de código
3. Cada sección debe ser independiente y ejecutable por separado

**Resultado esperado**: Notebook estructurado y listo para trabajar

---

## FASE 1: Exploración del Dataset Fashion-MNIST (45-60 min)

### Paso 1.1: Descargar y cargar el dataset

**Objetivo**: Cargar Fashion-MNIST con transformaciones apropiadas

**Descripción**:
- Definir transformaciones: `ToTensor()` y `Normalize((0.5,), (0.5,))`
- Descargar conjunto de entrenamiento (60,000 imágenes)
- Descargar conjunto de validación/test (10,000 imágenes)
- Verificar que los datos se descargaron correctamente

**Notas**:
- Los datos se guardarán en una carpeta local (ej: `MNIST_data/`)
- Las transformaciones normalizan los píxeles de [0,1] a [-1,1]

**Resultado esperado**: Datasets cargados y listos para usar

---

### Paso 1.2: Explorar el dataset

**Objetivo**: Entender la estructura y contenido del dataset

**Descripción**:
- Inspeccionar tamaño de los conjuntos (train: 60k, valid: 10k)
- Ver ejemplos de imágenes y sus etiquetas
- Verificar dimensiones de las imágenes (28x28, 1 canal)
- Crear diccionario de nombres de clases:
  - 0: T-shirt/top
  - 1: Trouser
  - 2: Pullover
  - 3: Dress
  - 4: Coat
  - 5: Sandal
  - 6: Shirt
  - 7: Sneaker
  - 8: Bag
  - 9: Ankle boots

**Resultado esperado**: Comprensión clara del formato de datos

---

### Paso 1.3: Visualización inicial

**Objetivo**: Crear visualizaciones para el informe

**Descripción**:
- Crear mosaico 3x3 de imágenes aleatorias con sus etiquetas
- Cada imagen debe mostrar su clase correspondiente
- Guardar figura en `images/fashion_mnist_ejemplos.png`
- Documentar el formato de los datos (tensores, dimensiones)

**Resultado esperado**: Figura guardada y lista para incluir en el informe

---

## FASE 2: Implementación de la Red Neuronal (60-90 min)

### Paso 2.1: Crear DataLoaders

**Objetivo**: Preparar los datos para entrenamiento en batches

**Descripción**:
- Crear `train_loader` con:
  - `batch_size = 100`
  - `shuffle = True`
- Crear `valid_loader` con:
  - `batch_size = 100`
  - `shuffle = True`
- Explorar un batch para entender la estructura:
  - Verificar dimensiones: `(batch_size, 1, 28, 28)` para imágenes
  - Verificar dimensiones: `(batch_size,)` para etiquetas

**Resultado esperado**: DataLoaders creados y funcionando

---

### Paso 2.2: Definir la arquitectura de la red

**Objetivo**: Implementar la red feedforward según especificaciones

**Descripción de la arquitectura**:
- **Capa de entrada**: Flatten de 28×28 = 784 neuronas
- **Capa oculta 1**: 
  - 128 neuronas
  - Función de activación: ReLU
  - Dropout: p=0.2
- **Capa oculta 2**:
  - 64 neuronas
  - Función de activación: ReLU
  - Dropout: p=0.2
- **Capa de salida**:
  - 10 neuronas (una por cada clase)
  - Sin función de activación (CrossEntropyLoss aplica softmax automáticamente)
  - Sin dropout

**Implementación**:
- Crear clase que hereda de `nn.Module`
- Definir capas en `__init__()`
- Implementar método `forward()` que define el flujo de datos

**Resultado esperado**: Clase de red neuronal definida

---

### Paso 2.3: Verificar la arquitectura

**Objetivo**: Asegurar que la red funciona correctamente

**Descripción**:
- Crear instancia del modelo
- Probar con un batch de ejemplo
- Verificar que la salida tenga dimensiones correctas: `(batch_size, 10)`
- Verificar que los valores de salida sean razonables

**Resultado esperado**: Red verificada y lista para entrenar

---

## FASE 3: Entrenamiento Básico (90-120 min)

### Paso 3.1: Configurar entrenamiento

**Objetivo**: Preparar todos los componentes necesarios para entrenar

**Descripción**:
- **Función de pérdida**: `nn.CrossEntropyLoss()`
  - Aplica automáticamente log_softmax
  - Compatible con etiquetas como enteros (0-9)
- **Optimizador**: `torch.optim.SGD(model.parameters(), lr=0.001)`
  - Learning rate inicial: 0.001
- **Dispositivo**: Detectar automáticamente (GPU si está disponible, sino CPU)
  - `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- Mover modelo al dispositivo: `model.to(device)`

**Resultado esperado**: Todo configurado para comenzar entrenamiento

---

### Paso 3.2: Implementar función de entrenamiento

**Objetivo**: Crear función que entrena el modelo por una época

**Descripción de `train_epoch()`**:
- Poner modelo en modo entrenamiento: `model.train()`
- Inicializar contadores para pérdida y precisión
- Iterar sobre `train_loader`:
  - Mover datos al dispositivo
  - Forward pass: calcular predicciones
  - Calcular pérdida
  - Backward pass: calcular gradientes
  - Actualizar pesos con optimizador
  - Acumular pérdida y calcular precisión
- Retornar pérdida promedio y precisión de la época

**Notas importantes**:
- Limpiar gradientes antes de cada batch: `optimizer.zero_grad()`
- Calcular precisión comparando predicciones con etiquetas reales

**Resultado esperado**: Función de entrenamiento implementada

---

### Paso 3.3: Implementar función de validación

**Objetivo**: Crear función que evalúa el modelo sin entrenar

**Descripción de `validate()`**:
- Poner modelo en modo evaluación: `model.eval()`
- Desactivar cálculo de gradientes: `torch.no_grad()`
- Inicializar contadores para pérdida y precisión
- Iterar sobre `valid_loader`:
  - Mover datos al dispositivo
  - Forward pass: calcular predicciones
  - Calcular pérdida
  - Acumular pérdida y calcular precisión
- Retornar pérdida promedio y precisión

**Notas importantes**:
- No actualizar pesos durante validación
- `torch.no_grad()` ahorra memoria y acelera el proceso

**Resultado esperado**: Función de validación implementada

---

### Paso 3.4: Loop principal de entrenamiento

**Objetivo**: Entrenar el modelo por múltiples épocas

**Descripción**:
- Inicializar listas para guardar métricas:
  - `train_losses = []`
  - `train_accuracies = []`
  - `val_losses = []`
  - `val_accuracies = []`
- Loop sobre épocas (empezar con 10-15 épocas):
  - Llamar a `train_epoch()` y guardar resultados
  - Llamar a `validate()` y guardar resultados
  - Opcional: imprimir progreso cada época
- Guardar todas las métricas en las listas

**Notas importantes**:
- No olvidar mover los datos al dispositivo en cada iteración
- Guardar métricas para poder graficar después

**Resultado esperado**: Modelo entrenado y métricas guardadas

---

### Paso 3.5: Primer entrenamiento

**Objetivo**: Ejecutar el primer entrenamiento completo

**Descripción**:
- Ejecutar el loop de entrenamiento
- Verificar que las métricas se calculen correctamente
- Observar el comportamiento inicial:
  - ¿La pérdida disminuye?
  - ¿La precisión aumenta?
  - ¿Hay diferencia entre train y validation?

**Resultado esperado**: Modelo entrenado con resultados iniciales

---

## FASE 4: Visualización de Resultados Básicos (45-60 min)

### Paso 4.1: Gráficos de curvas de entrenamiento

**Objetivo**: Visualizar el progreso del entrenamiento

**Descripción**:
- Crear figura con 2 subplots:
  - **Subplot 1**: Pérdida vs épocas
    - Línea para entrenamiento
    - Línea para validación
    - Leyenda, etiquetas de ejes, título
  - **Subplot 2**: Precisión vs épocas
    - Línea para entrenamiento
    - Línea para validación
    - Leyenda, etiquetas de ejes, título
- Guardar en `images/curvas_entrenamiento_basico.png`

**Análisis a realizar**:
- ¿Cuántas épocas son necesarias?
- ¿Hay sobreajuste? (train mejora pero validation se estanca)
- ¿Hay subajuste? (ambas mejoran pero podrían mejorar más)

**Resultado esperado**: Figura guardada con curvas de entrenamiento

---

### Paso 4.2: Matriz de confusión

**Objetivo**: Analizar el rendimiento por clase

**Descripción**:
- Evaluar modelo completo sobre conjunto de validación
- Obtener todas las predicciones y etiquetas reales
- Calcular matriz de confusión usando `sklearn.metrics.confusion_matrix`
- Visualizar con `matplotlib` o `seaborn`:
  - Heatmap con colores
  - Etiquetas de clases en ejes
  - Valores numéricos en cada celda
  - Título y etiquetas descriptivas
- Guardar en `images/matriz_confusion_basico.png`

**Análisis a realizar**:
- ¿Qué clases se confunden más?
- ¿Hay clases con mejor/peor rendimiento?
- Precisión por clase

**Resultado esperado**: Matriz de confusión generada y guardada

---

### Paso 4.3: Análisis inicial

**Objetivo**: Identificar puntos clave para el informe

**Descripción**:
- Determinar número óptimo de épocas:
  - Buscar punto donde validation loss deja de mejorar
  - Evitar sobreajuste
- Observar comportamiento train vs validation:
  - ¿Se separan las curvas? (sobreajuste)
  - ¿Van juntas? (buen ajuste)
- Anotar observaciones para la discusión del informe

**Resultado esperado**: Análisis preliminar completado

---

## FASE 5: Análisis de Hiperparámetros (Mínimo Viable) (120-150 min)

### Paso 5.1: Variar Learning Rate

**Objetivo**: Encontrar el learning rate óptimo

**Descripción**:
- Probar 3 valores de learning rate:
  - `lr = 0.0001` (muy bajo)
  - `lr = 0.001` (valor inicial)
  - `lr = 0.01` (alto)
- Para cada valor:
  - Reinicializar el modelo (mismos pesos iniciales)
  - Entrenar 10-15 épocas
  - Guardar curvas de entrenamiento
- Comparar resultados:
  - ¿Cuál converge más rápido?
  - ¿Cuál alcanza mejor precisión?
  - ¿Algún valor causa inestabilidad?

**Visualización**:
- Crear figura comparativa con curvas de los 3 learning rates
- Guardar en `images/comparacion_learning_rates.png`

**Resultado esperado**: Learning rate óptimo identificado

---

### Paso 5.2: Comparar Optimizadores

**Objetivo**: Comparar SGD vs ADAM

**Descripción**:
- Probar dos optimizadores con mismo learning rate (0.001):
  - **SGD**: `torch.optim.SGD(model.parameters(), lr=0.001)`
  - **ADAM**: `torch.optim.Adam(model.parameters(), lr=0.001)`
- Para cada optimizador:
  - Reinicializar el modelo
  - Entrenar 10-15 épocas
  - Guardar curvas de entrenamiento
  - Medir tiempo de entrenamiento
- Comparar resultados:
  - ¿Cuál converge más rápido?
  - ¿Cuál alcanza mejor precisión final?
  - ¿Cuál es más estable?

**Visualización**:
- Crear figura comparativa con curvas de ambos optimizadores
- Guardar en `images/comparacion_optimizadores.png`

**Resultado esperado**: Optimizador óptimo identificado

---

### Paso 5.3: Variar Dropout (Opcional - si hay tiempo)

**Objetivo**: Analizar efecto de la regularización

**Descripción**:
- Probar 3 valores de dropout:
  - `dropout = 0.0` (sin regularización)
  - `dropout = 0.2` (valor inicial)
  - `dropout = 0.5` (alta regularización)
- Para cada valor:
  - Modificar arquitectura de la red
  - Entrenar con mismos hiperparámetros
  - Observar diferencia entre train y validation
- Comparar resultados:
  - ¿Cuál tiene menor sobreajuste?
  - ¿Cuál tiene mejor precisión en validation?

**Nota**: Este paso es opcional si el tiempo es limitado

**Resultado esperado**: (Opcional) Efecto del dropout analizado

---

### Paso 5.4: Generar figuras comparativas

**Objetivo**: Crear visualizaciones para el informe

**Descripción**:
- Revisar todas las figuras generadas
- Asegurar que tengan:
  - Títulos descriptivos
  - Etiquetas en ejes
  - Leyendas claras
  - Formato adecuado para incluir en PDF
- Organizar en carpeta `images/` con nombres descriptivos

**Resultado esperado**: Todas las figuras listas para el informe

---

## FASE 6: Generación de Figuras Finales (30-45 min)

### Paso 6.1: Entrenar modelo final

**Objetivo**: Entrenar modelo con mejores hiperparámetros encontrados

**Descripción**:
- Seleccionar mejores hiperparámetros basado en análisis previo:
  - Mejor learning rate
  - Mejor optimizador
  - Dropout (0.2 o el mejor encontrado)
- Reinicializar modelo
- Entrenar por número óptimo de épocas (determinado en Fase 4)
- Guardar modelo entrenado (opcional)

**Resultado esperado**: Modelo final entrenado

---

### Paso 6.2: Generar figuras finales

**Objetivo**: Crear figuras con resultados finales

**Descripción**:
- **Curvas de entrenamiento finales**:
  - Usar modelo final entrenado
  - Guardar en `images/curvas_entrenamiento_final.png`
- **Matriz de confusión final**:
  - Evaluar modelo final sobre validation
  - Guardar en `images/matriz_confusion_final.png`
- Asegurar calidad y claridad de las figuras

**Resultado esperado**: Figuras finales de alta calidad

---

### Paso 6.3: Organizar figuras

**Objetivo**: Tener todas las figuras listas para el informe

**Descripción**:
- Listar todas las figuras generadas:
  - `fashion_mnist_ejemplos.png`
  - `curvas_entrenamiento_basico.png`
  - `matriz_confusion_basico.png`
  - `comparacion_learning_rates.png`
  - `comparacion_optimizadores.png`
  - `curvas_entrenamiento_final.png`
  - `matriz_confusion_final.png`
- Verificar que todas estén en `images/`
- Anotar nombres para referenciar en el informe

**Resultado esperado**: Figuras organizadas y listas

---

## FASE 7: Redacción del Informe (120-180 min)

### Paso 7.1: Título y Autores

**Archivo**: `informe_cientifico/01_titulo_autores.md`

**Contenido**:
- Título: "Clasificación de Fashion-MNIST con una Red Neuronal Feedforward Multicapa en PyTorch"
- Nombre completo de todos los integrantes del grupo
- Correos electrónicos
- Institución (FAMAFyC)

**Resultado esperado**: Sección de título y autores completada

---

### Paso 7.2: Resumen

**Archivo**: `informe_cientifico/02_resumen.md`

**Contenido** (5-10 líneas):
- Objetivo del trabajo
- Metodología empleada (red feedforward, PyTorch, Fashion-MNIST)
- Principales resultados obtenidos:
  - Precisión alcanzada
  - Hiperparámetros óptimos encontrados
  - Conclusiones principales

**Resultado esperado**: Resumen conciso y completo

---

### Paso 7.3: Introducción

**Archivo**: `informe_cientifico/03_introduccion.md`

**Contenido**:
- Contexto general:
  - ¿Qué es una red neuronal artificial feedforward multicapa?
  - ¿Cómo se usa para clasificar datos?
- Dataset Fashion-MNIST:
  - Breve descripción
  - Es un reemplazo directo (drop-in replacement) de MNIST con el mismo formato (28×28, escala de grises, 10 clases, 70,000 imágenes)
  - Más desafiante que MNIST: mientras MNIST puede alcanzar >99% de precisión, Fashion-MNIST presenta un desafío mayor para algoritmos de machine learning
  - Diseñado específicamente como mejor benchmark que MNIST
  - Por qué es relevante para este trabajo
- Objetivo del informe:
  - Implementar red feedforward
  - Clasificar Fashion-MNIST
  - Analizar hiperparámetros
- Bibliografía relevante:
  - Documentación PyTorch
  - Repositorio Fashion-MNIST
  - Referencias sobre redes neuronales

**Resultado esperado**: Introducción completa con contexto y objetivos claros

---

### Paso 7.4: Teoría

**Archivo**: `informe_cientifico/04_teoria.md`

**Contenido**:

**Dataset Fashion-MNIST**:
- Características: 70,000 imágenes, 28×28 píxeles, escala de grises
- 10 clases de prendas de vestir
- División: 60,000 entrenamiento, 10,000 validación
- Normalización aplicada
- **Compatibilidad**: Drop-in replacement de MNIST, mismo formato de archivos (compatible con todas las librerías que soportan MNIST)
- **Desafío**: Más difícil que MNIST (clasificar prendas de moda vs dígitos escritos a mano)
- Basado en productos reales de moda de Zalando

**Arquitectura de la Red Neuronal**:
- Estructura detallada:
  - Capa de entrada: 784 neuronas (28×28 aplanado)
  - Capa oculta 1: 128 neuronas + ReLU + Dropout(0.2)
  - Capa oculta 2: 64 neuronas + ReLU + Dropout(0.2)
  - Capa de salida: 10 neuronas
- Justificación de decisiones de diseño

**Función de Pérdida y Optimización**:
- Cross Entropy Loss:
  - Explicar qué es y por qué se usa
  - Mencionar que aplica softmax automáticamente
- Optimizadores:
  - SGD (Stochastic Gradient Descent)
  - ADAM (Adaptive Moment Estimation)
  - Comparación breve

**Proceso de Entrenamiento y Validación**:
- Descripción del loop de entrenamiento
- División train/validation
- Métricas utilizadas (pérdida y precisión)

**Resultado esperado**: Sección teórica completa y detallada

---

### Paso 7.5: Resultados

**Archivo**: `informe_cientifico/05_resultados.md`

**Contenido**:

**Resultados Básicos**:
- Describir figura `curvas_entrenamiento_basico.png`:
  - Comportamiento de pérdida y precisión
  - Número óptimo de épocas identificado
  - Observaciones sobre train vs validation
- Describir figura `matriz_confusion_basico.png`:
  - Precisión general
  - Clases con mejor/peor rendimiento
  - Confusiones más comunes

**Análisis de Hiperparámetros**:
- **Learning Rate**:
  - Describir figura `comparacion_learning_rates.png`
  - Comparar los 3 valores probados
  - Identificar el óptimo y justificar
- **Optimizadores**:
  - Describir figura `comparacion_optimizadores.png`
  - Comparar SGD vs ADAM
  - Identificar el mejor y justificar

**Resultados Finales**:
- Describir figura `curvas_entrenamiento_final.png`:
  - Curvas con mejores hiperparámetros
  - Precisión final alcanzada
- Describir figura `matriz_confusion_final.png`:
  - Rendimiento final por clase
  - Precisión general final

**Tablas (si aplica)**:
- Tabla comparativa de hiperparámetros
- Tabla de precisión por clase

**Importante**: 
- Cada figura debe ser mencionada en el texto
- Incluir descripciones detalladas
- Etiquetas y unidades en todas las figuras

**Resultado esperado**: Sección de resultados completa con todas las figuras referenciadas

---

### Paso 7.6: Discusión y Conclusiones

**Archivo**: `informe_cientifico/06_discusion.md`

**Contenido**:

**Análisis del Rendimiento**:
- Precisión alcanzada: ¿es buena? ¿qué significa?
  - **Contexto importante**: Fashion-MNIST es más desafiante que MNIST
  - Mientras MNIST puede alcanzar >99% de precisión con redes feedforward, Fashion-MNIST típicamente alcanza 85-90% con arquitecturas similares
  - Una precisión en el rango 85-90% es razonable y esperada para este problema
  - Comparar con benchmarks del paper si es relevante
- Comportamiento de curvas:
  - ¿Hubo sobreajuste? ¿cómo se manifestó?
  - ¿Hubo subajuste? ¿qué se podría mejorar?
- Comparación train vs validation

**Efecto de los Hiperparámetros**:
- **Learning Rate**:
  - ¿Cómo afectó al entrenamiento?
  - ¿Por qué el valor óptimo funcionó mejor?
- **Optimizador**:
  - ¿Qué diferencias se observaron?
  - ¿Por qué uno fue mejor que el otro?

**Limitaciones del Enfoque**:
- ¿Qué limitaciones tiene una red feedforward para este problema?
- ¿Qué mejoras se podrían implementar?
- Arquitecturas alternativas (convolucionales, etc.)

**Mejoras Futuras**:
- Propuestas concretas:
  - Arquitecturas más complejas
  - Técnicas de regularización adicionales
  - Preprocesamiento de datos
  - Data augmentation

**Conclusiones**:
- Resumir logros principales
- Confirmar si el objetivo se cumplió
- Reflexiones finales

**Resultado esperado**: Discusión completa con análisis crítico

---

### Paso 7.7: Referencias

**Archivo**: `informe_cientifico/07_referencias.md`

**Contenido**:
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PyTorch Tutorials: https://pytorch.org/tutorials/
- PyTorch Optimization Tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Fashion-MNIST Repository: https://github.com/zalandoresearch/fashion-mnist
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747.
- (Agregar otras referencias utilizadas)

**Formato**: Lista numerada o con viñetas, formato coherente

**Resultado esperado**: Referencias completas y correctamente formateadas

---

## FASE 8: Compilación y Revisión Final (30-45 min)

### Paso 8.1: Compilar informe completo

**Objetivo**: Unir todas las secciones en un solo archivo

**Método usando pandoc** (como en TP1):
```bash
cd informe_cientifico
pandoc 01_titulo_autores.md 02_resumen.md 03_introduccion.md 04_teoria.md 05_resultados.md 06_discusion.md 07_referencias.md -o informe_completo.pdf
```

**Alternativa - Unir en markdown primero**:
```bash
cat 01_titulo_autores.md 02_resumen.md 03_introduccion.md 04_teoria.md 05_resultados.md 06_discusion.md 07_referencias.md > informe_completo.md
pandoc informe_completo.md -o informe_completo.pdf
```

**Verificaciones**:
- Todas las secciones están incluidas
- Las figuras se referencian correctamente
- El formato es consistente

**Resultado esperado**: PDF generado correctamente

---

### Paso 8.2: Verificar formato y extensión

**Objetivo**: Asegurar que cumple con los requisitos

**Verificaciones**:
- **Extensión**: No exceder 4 páginas
- **Formato**: PDF
- **Nombre del archivo**: `grupo-número_apellido1_apellido2_apellido3_TP2_RN_2025.pdf`
- **Calidad de figuras**: Todas se ven bien en el PDF
- **Referencias**: Todas las figuras están mencionadas

**Ajustes si es necesario**:
- Reducir tamaño de figuras si el PDF es muy largo
- Ajustar márgenes
- Optimizar espacio en texto

**Resultado esperado**: PDF final que cumple todos los requisitos

---

### Paso 8.3: Revisión final

**Objetivo**: Última revisión antes de entregar

**Checklist**:
- [ ] Todas las secciones completas
- [ ] Figuras incluidas y referenciadas
- [ ] Ortografía y gramática revisadas
- [ ] Formato de nombre correcto
- [ ] Extensión ≤ 4 páginas
- [ ] Referencias completas
- [ ] Código en notebook funciona correctamente
- [ ] Todas las figuras guardadas en `images/`

**Resultado esperado**: Trabajo listo para entregar

---

## Resumen de Entregables

### Archivos de código:
- `tp2_trabajo.ipynb` - Notebook con todo el código implementado

### Figuras (en `images/`):
- `fashion_mnist_ejemplos.png`
- `curvas_entrenamiento_basico.png`
- `matriz_confusion_basico.png`
- `comparacion_learning_rates.png`
- `comparacion_optimizadores.png`
- `curvas_entrenamiento_final.png`
- `matriz_confusion_final.png`

### Informe:
- `informe_cientifico/informe_completo.pdf` - PDF final para entregar
- Archivos markdown individuales en `informe_cientifico/`

---

## Tiempo Estimado Total

- **Fase 0**: 60-90 min (Instalación)
- **Fase 1**: 45-60 min (Exploración dataset)
- **Fase 2**: 60-90 min (Implementación red)
- **Fase 3**: 90-120 min (Entrenamiento básico)
- **Fase 4**: 45-60 min (Visualización básica)
- **Fase 5**: 120-150 min (Análisis hiperparámetros)
- **Fase 6**: 30-45 min (Figuras finales)
- **Fase 7**: 120-180 min (Redacción informe)
- **Fase 8**: 30-45 min (Compilación)

**Total estimado**: 600-840 minutos (10-14 horas)

---

## Notas Importantes

1. **Trabajo incremental**: Cada fase construye sobre la anterior
2. **Guardar frecuentemente**: Guardar el notebook y las figuras regularmente
3. **Documentar decisiones**: Anotar por qué se eligieron ciertos valores
4. **Debugging**: Usar celdas individuales para probar y depurar
5. **Figuras**: Asegurar calidad y claridad desde el inicio
6. **Tiempo**: Este es un plan mínimo viable, se puede expandir después

---

## Recursos de Ayuda

- Documentación PyTorch: https://pytorch.org/docs/stable/index.html
- Tutoriales PyTorch: https://pytorch.org/tutorials/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Bibliografía en `bibliografia/`: Archivos markdown con conceptos clave

---

**¡Éxito con el trabajo práctico!**

