## Teoría

### Arquitectura de la red neuronal

La arquitectura central empleada es una red neuronal feedforward (multilayer perceptron) con capas densas (fully connected). Se definen variantes para experimentar con número de neuronas y valores de dropout.
Estructura base (clase `FashionMNIST_Net`):

- Capa de entrada: Flatten de la imagen 28 × 28 → vector de 784 características.
- Capa oculta 1 (FC1): 128 neuronas + ReLU + Dropout(0.2).
- Capa oculta 2 (FC2): 64 neuronas + ReLU + Dropout(0.2).
- Capa de salida (FC3): 10 neuronas (una por clase). La salida se combina con la función de pérdida CrossEntropyLoss (PyTorch), que aplica softmax internamente.

### Arquitectura Feedforward
- Redes donde la información fluye en una sola dirección: entrada → capas ocultas → salida
- Sin ciclos o retroalimentación
- También conocidas como Multi-Layer Perceptron (MLP)

### Componentes Principales

#### Capas Lineales (Fully Connected)
- Cada neurona está conectada a todas las neuronas de la capa siguiente
- Operación: `y = Wx + b`
- Donde W es la matriz de pesos y b es el vector de sesgos

#### Funciones de Activación
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`
  - Muy común en capas ocultas
  - Soluciona el problema de gradientes que desaparecen
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))`
  - Útil para salidas en [0, 1]
- **Softmax**: Para clasificación multiclase
  - Normaliza salidas a probabilidades que suman 1

#### Dropout
- Técnica de regularización
- Durante el entrenamiento, desactiva aleatoriamente un porcentaje de neuronas
- Previene sobreajuste
- Típicamente p = 0.2 a 0.5

### Backpropagation
- Algoritmo para calcular gradientes
- Propaga el error desde la salida hacia atrás
- Permite actualizar los pesos mediante descenso de gradiente

## Hiperparámetros Importantes

### Arquitectura
- Número de capas ocultas
- Número de neuronas por capa
- Tipo de función de activación

### Entrenamiento
- Learning rate: controla el tamaño del paso en la optimización
- Batch size: número de ejemplos por iteración
- Número de épocas: cuántas veces se recorre el dataset completo
- Optimizador: SGD, ADAM, etc.

### Regularización
- Dropout: probabilidad de desactivar neuronas
- Weight decay: penalización L2 sobre los pesos
- Early stopping: detener cuando la validación deja de mejorar

### Arquitectura sugerida
- Entrada: 784 neuronas (28×28 píxeles aplanados)
- Capas ocultas: típicamente 128-512 neuronas
- Salida: 10 neuronas (una por cada clase)
- Activación: ReLU en capas ocultas, ninguna en salida (CrossEntropyLoss aplica softmax)

### Desafíos
- Imágenes pequeñas (28×28) pero con suficiente información
- 10 clases balanceadas
- Buena tarea para aprender conceptos básicos de deep learning


## Definición de Hiperparámetros

### 1. Learning Rate (Tasa de Aprendizaje)

**¿Qué es?**
- El learning rate controla qué tan grandes son los pasos que da el optimizador al actualizar los pesos de la red.
- Es un multiplicador que determina cuánto cambian los pesos en cada iteración.

**Analogía:**
- Si estás bajando una montaña, el learning rate es el tamaño de tus pasos:
  - **Muy alto (0.01)**: Pasos grandes, puedes pasar por encima del mínimo o oscilar.
  - **Muy bajo (0.0001)**: Pasos pequeños, avance muy lento, puede quedar atascado.
  - **Óptimo (0.001)**: Pasos moderados, convergencia más estable.

**Por qué importa:**
- Afecta directamente la velocidad de aprendizaje y la estabilidad del entrenamiento.
- Valores probados: `[0.0001, 0.001, 0.01]`

---

### 2. Optimizador

**¿Qué es?**
- El optimizador es el algoritmo que actualiza los pesos de la red para minimizar la pérdida.
- Comparamos dos optimizadores: **SGD** y **ADAM**.

**SGD (Stochastic Gradient Descent):**
- Actualiza los pesos en la dirección opuesta al gradiente.
- Simple pero puede ser lento y quedar atascado en mínimos locales.
- No tiene memoria de pasos anteriores.

**ADAM (Adaptive Moment Estimation):**
- Adapta el learning rate por parámetro.
- Usa promedios móviles de gradientes (momentum) y de sus cuadrados.
- Suele converger más rápido y ser más robusto.

**Por qué importa:**
- El optimizador determina cómo se actualizan los pesos, influyendo en la velocidad y calidad de la convergencia.

---

### 3. Dropout

**¿Qué es?**
- Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento.
- El valor de dropout (p) es la probabilidad de que una neurona se desactive.

**Cómo funciona:**
- Durante el entrenamiento: cada neurona tiene probabilidad `p` de desactivarse.
- Durante la validación: todas las neuronas están activas, pero sus salidas se escalan por `(1-p)`.

**Por qué importa:**
- Reduce el overfitting al evitar que la red dependa demasiado de neuronas específicas.
- Valores probados: `[0.0, 0.2, 0.4, 0.6]`
  - **0.0**: Sin dropout (más riesgo de overfitting)
  - **0.2**: Moderado (valor común)
  - **0.4-0.6**: Alto (puede subentrenar si es excesivo)

---

### 4. Número de Neuronas en Capas Intermedias

**¿Qué es?**
- Cantidad de neuronas en cada capa oculta de la red.
- En nuestro caso: primera capa oculta (n1) y segunda capa oculta (n2).

**Capacidad del modelo:**
- **Más neuronas**: Más capacidad, puede aprender patrones más complejos, pero más riesgo de overfitting y más lento.
- **Menos neuronas**: Menos capacidad, más rápido, pero puede no capturar suficiente complejidad.

**Configuraciones probadas:**
- `(64, 32)`: Pequeña
- `(128, 64)`: Original (baseline)
- `(256, 128)`: Grande
- `(512, 256)`: Muy grande

**Por qué importa:**
- Determina la capacidad de aprendizaje del modelo y el balance entre complejidad y generalización.

---

### 5. Número de Épocas

**¿Qué es?**
- Cantidad de veces que el modelo ve todo el conjunto de entrenamiento completo.

**Cómo funciona:**
- Una época = una pasada completa por todos los datos de entrenamiento.
- Más épocas = más oportunidades de aprender, pero riesgo de overfitting si se entrena demasiado.

**Por qué importa:**
- Determina cuánto tiempo se entrena el modelo.
- Valores probados: `[5, 10, 15, 20, 30]`
  - **Pocas épocas**: Puede subentrenar
  - **Demasiadas**: Puede sobreentrenar
  - **Óptimo**: Cuando la precisión de validación deja de mejorar

---

### 6. Batch Size (Tamaño del Lote)

**¿Qué es?**
- Número de ejemplos que el modelo procesa antes de actualizar los pesos.

**Cómo funciona:**
- **Batch size pequeño (32)**: Actualizaciones más frecuentes, más ruido, más lento por época.
- **Batch size grande (256)**: Actualizaciones menos frecuentes, gradientes más estables, más rápido por época.

**Trade-offs:**
- **Pequeño**: Más exploración, más lento, más memoria por actualización.
- **Grande**: Más estable, más rápido, pero puede quedar atascado en mínimos locales.

**Valores probados:** `[32, 64, 100, 128, 256]`

**Por qué importa:**
- Afecta la estabilidad del entrenamiento, la velocidad y el uso de memoria.