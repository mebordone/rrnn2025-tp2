## Metodología

### Arquitectura de la red neuronal

Se implementó una red neuronal feedforward (multilayer perceptron) con capas densas (fully connected). La estructura base consiste en:

- **Capa de entrada**: Flatten de la imagen 28×28 → vector de 784 características
- **Capa oculta 1**: 128 neuronas + ReLU + Dropout(0.2)
- **Capa oculta 2**: 64 neuronas + ReLU + Dropout(0.2)
- **Capa de salida**: 10 neuronas (una por clase) con CrossEntropyLoss

### Hiperparámetros evaluados

Se realizó una búsqueda sistemática de hiperparámetros variando:
- **Learning rate**: [0.0001, 0.001, 0.01]
- **Optimizador**: SGD y ADAM
- **Dropout**: [0.0, 0.2, 0.4, 0.6]
- **Arquitectura**: (64,32), (128,64), (256,128), (512,256)
- **Épocas**: [5, 10, 15, 20, 30]
- **Batch size**: [32, 64, 100, 128, 256]

