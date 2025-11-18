## Introducción

El objetivo principal de este informe es implementar, entrenar y analizar un clasificador para el dataset Fashion-MNIST utilizando una Red Neuronal Artificial Feedforward Multicapa en el entorno PyTorch.

Fashion-MNIST es un dataset estándar en el campo del aprendizaje automático, diseñado como reemplazo del clásico MNIST pero con una mayor dificultad. Contiene 70.000 imágenes en escala de grises divididas en 10 categorías de artículos de moda.
- Contenido: 70.000 imágenes en escala de grises.
- Dimensiones: 28 × 28 píxeles por imagen.
- Clases: 10 clases mutuamente excluyentes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- División: 60.000 imágenes para entrenamiento y 10.000 para validación/prueba.
El conjunto de entrenamiento se utiliza para ajustar los parámetros del modelo (pesos y sesgos). El conjunto de validación (o test) se emplea para estimar la capacidad de generalización del modelo y para la selección de hiperparámetros, ayudando a detectar sobreajuste (overfitting). 

Como objetivos de este informe proponemos:

Explorar el conjunto de datos y establecer una arquitectura de red base.

Evaluar el impacto de distintos hiperparámetros (tasas de aprendizaje, optimizadores, dropout, tamaños de batch) en el rendimiento del modelo.

Seleccionar y entrenar la configuración óptima para obtener la máxima precisión de clasificación.

