# Resultados de Experimentos - TP2 Fashion-MNIST

**Fecha de ejecución**: 2025-11-17 14:05:34

---

## Tabla de Contenidos

1. [Configuración del Entorno](#configuración-del-entorno)
2. [Dataset Fashion-MNIST](#dataset-fashion-mnist)
3. [Entrenamiento Básico](#entrenamiento-básico)
4. [Análisis de Hiperparámetros](#análisis-de-hiperparámetros)
5. [Modelo Final](#modelo-final)

---


## Configuración del Entorno

- **Dispositivo**: cuda
- **GPU**: NVIDIA GeForce RTX 3060
- **CUDA Version**: 12.8
- **PyTorch Version**: 2.9.1+cu128


## Dataset Fashion-MNIST

- **Tamaño conjunto de entrenamiento**: 60000
- **Tamaño conjunto de validación**: 10000
- **Dimensiones de imagen**: torch.Size([1, 28, 28])
- **Número de clases**: 10

**Mosaico 3x3 de ejemplos del dataset Fashion-MNIST**

![Mosaico 3x3 de ejemplos del dataset Fashion-MNIST](images/fashion_mnist_ejemplos.png)


## Entrenamiento Básico

### Configuración
- **Optimizador**: SGD
- **Learning Rate**: 0.001
- **Batch Size**: 100
- **Épocas**: 30
- **Dropout**: 0.2

### Resultados Finales
- **Train Accuracy**: 77.49%
- **Validation Accuracy**: 78.97%
- **Train Loss**: 0.6215
- **Validation Loss**: 0.5697

**Curvas de entrenamiento básico**

![Curvas de entrenamiento básico](images/curvas_entrenamiento_basico.png)

**Matriz de confusión - Entrenamiento básico**

![Matriz de confusión - Entrenamiento básico](images/matriz_confusion_basico.png)

**Precisión general en validación**: 78.97%


## Análisis de Hiperparámetros


### Experimento 1: Variar Learning Rate

**Valores probados**: 0.0001, 0.001, 0.01
**Configuración**: SGD, Batch Size=100, Épocas=10, Dropout=0.2

| Learning Rate | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| 0.0001 | 33.12% | 45.26% |
| 0.001 | 68.48% | 73.47% |
| 0.01 | 83.64% | 83.66% |

**Comparación de Learning Rates**

![Comparación de Learning Rates](images/comparacion_learning_rates.png)



### Experimento 2: Comparar Optimizadores (SGD vs ADAM)

**Optimizadores probados**: SGD, ADAM
**Configuración**: LR=0.001, Batch Size=100, Épocas=10, Dropout=0.2

| Optimizador | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| SGD | 67.93% | 74.35% |
| ADAM | 88.56% | 87.38% |

**Comparación de Optimizadores**

![Comparación de Optimizadores](images/comparacion_optimizadores.png)



### Experimento 3: Variar Dropout

**Valores probados**: 0.0, 0.2, 0.4, 0.6
**Configuración**: SGD, LR=0.001, Batch Size=100, Épocas=10

| Dropout | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| 0.0 | 75.06% | 74.76% |
| 0.2 | 68.78% | 73.69% |
| 0.4 | 62.57% | 72.44% |
| 0.6 | 53.20% | 71.01% |

**Comparación de Dropout**

![Comparación de Dropout](images/comparacion_dropout.png)



### Experimento 4: Variar Número de Neuronas

**Configuraciones probadas**: (64,32), (128,64), (256,128), (512,256)
**Configuración**: SGD, LR=0.001, Batch Size=100, Épocas=10, Dropout=0.2

| Arquitectura | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| 64-32 | 65.15% | 73.58% |
| 128-64 | 68.22% | 73.79% |
| 256-128 | 71.05% | 74.09% |
| 512-256 | 73.36% | 75.34% |

**Comparación de Arquitecturas**

![Comparación de Arquitecturas](images/comparacion_neuronas.png)



### Experimento 5: Variar Número de Épocas

**Valores probados**: 5, 10, 15, 20, 30
**Configuración**: SGD, LR=0.001, Batch Size=100, Dropout=0.2

| Épocas | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| 5 | 60.97% | 70.56% |
| 10 | 68.13% | 72.86% |
| 15 | 72.47% | 75.38% |
| 20 | 74.72% | 77.12% |
| 30 | 77.46% | 79.19% |

**Comparación de Épocas**

![Comparación de Épocas](images/comparacion_epocas.png)



### Experimento 6: Variar Batch Size

**Valores probados**: 32, 64, 100, 128, 256
**Configuración**: SGD, LR=0.001, Épocas=10, Dropout=0.2

| Batch Size | Train Accuracy | Validation Accuracy |
| --- | --- | --- |
| 32 | 78.09% | 79.63% |
| 64 | 71.66% | 74.93% |
| 100 | 68.05% | 72.93% |
| 128 | 65.51% | 72.44% |
| 256 | 55.79% | 66.08% |

**Comparación de Batch Size**

![Comparación de Batch Size](images/comparacion_batch_size.png)



## Modelo Final

### Configuración Óptima Aplicada
- **Optimizador**: ADAM
- **Learning Rate**: 0.001
- **Arquitectura**: 256-128
- **Batch Size**: 32
- **Épocas**: 30
- **Dropout**: 0.2

**Curvas de entrenamiento - Modelo Final**

![Curvas de entrenamiento - Modelo Final](images/curvas_entrenamiento_final.png)


**Matriz de confusión - Modelo Final**

![Matriz de confusión - Modelo Final](images/matriz_confusion_final.png)

**Precisión final en validación**: 87.78%

---

*Documento generado automáticamente el 2025-11-17 14:47:40*
