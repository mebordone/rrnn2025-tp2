# PyTorch: Fundamentos y Conceptos Clave

## Documentación Oficial

### Recursos principales:
- **Documentación PyTorch**: https://pytorch.org/docs/stable/index.html
- **Tutoriales PyTorch**: https://pytorch.org/tutorials/
- **Guía de optimización**: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

## Conceptos Clave

### Tensores
- Los tensores son la estructura de datos fundamental en PyTorch
- Similar a arrays de NumPy pero con soporte para GPU
- Operaciones automáticas de diferenciación

### Modelos (nn.Module)
- Clase base para todos los modelos de redes neuronales
- Define la arquitectura de la red
- Implementa métodos `forward()` para el paso hacia adelante

### DataLoader
- Utilidad para cargar datos en batches
- Soporte para shuffle, num_workers, etc.
- Facilita el entrenamiento eficiente

### Funciones de Pérdida (Loss Functions)
- **Cross Entropy Loss**: Comúnmente usada para clasificación multiclase
- Aplica automáticamente log_softmax
- Compatible con clases codificadas como enteros (no one-hot)

### Optimizadores
- **SGD (Stochastic Gradient Descent)**: Método clásico de optimización
- **ADAM**: Optimizador adaptativo, generalmente más eficiente
- Requieren parámetros del modelo y learning rate

### Entrenamiento
1. Forward pass: calcular predicciones
2. Calcular pérdida
3. Backward pass: calcular gradientes
4. Actualizar parámetros con optimizador

## Notas de Implementación

### Estructura típica de entrenamiento:
```python
# Definir modelo
model = MyNeuralNetwork()

# Definir función de pérdida
criterion = nn.CrossEntropyLoss()

# Definir optimizador
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Loop de entrenamiento
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Dispositivos (CPU/GPU)
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Mover modelo y datos al dispositivo: `model.to(device)`, `data.to(device)`

