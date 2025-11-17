# Redes Neuronales Feedforward Multicapa

## Conceptos Fundamentales

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

## Consideraciones para Fashion-MNIST

### Arquitectura sugerida
- Entrada: 784 neuronas (28×28 píxeles aplanados)
- Capas ocultas: típicamente 128-512 neuronas
- Salida: 10 neuronas (una por cada clase)
- Activación: ReLU en capas ocultas, ninguna en salida (CrossEntropyLoss aplica softmax)

### Desafíos
- Imágenes pequeñas (28×28) pero con suficiente información
- 10 clases balanceadas
- Buena tarea para aprender conceptos básicos de deep learning

