## Discusión y Conclusiones

Los resultados demuestran que la selección adecuada de hiperparámetros es crucial para el rendimiento del modelo. El optimizador ADAM mostró una ventaja significativa sobre SGD, alcanzando 88.06% de precisión en validación frente a 73.44% de SGD. La combinación óptima de hiperparámetros (ADAM, LR=0.001, arquitectura 256-128, batch size 32, dropout 0.2, 30 épocas) logró **87.99%** de precisión en validación.

**Configuración óptima recomendada:**
- Optimizador: ADAM
- Learning Rate: 0.001
- Arquitectura: 256-128
- Batch Size: 32
- Épocas: 30
- Dropout: 0.2

El análisis de la matriz de confusión revela que la categoría "Shirt" presenta la mayor dificultad de clasificación, posiblemente debido a su similitud visual con otras prendas como "T-shirt/top" y "Pullover". El modelo alcanza un rendimiento dentro del rango esperado para Fashion-MNIST (85-90%), demostrando un buen balance entre capacidad de aprendizaje y generalización.

