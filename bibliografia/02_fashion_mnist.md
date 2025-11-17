# Fashion-MNIST Dataset

## Información General

### Repositorio Oficial
- GitHub: https://github.com/zalandoresearch/fashion-mnist
- Paper: https://arxiv.org/abs/1708.07747

### Características del Dataset
- **Tamaño**: 70,000 imágenes en escala de grises
  - 60,000 para entrenamiento
  - 10,000 para test
- **Dimensiones**: 28x28 píxeles
- **Clases**: 10 categorías de prendas de vestir
- **Formato**: Similar a MNIST pero con imágenes de ropa

### Categorías (10 clases)
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Uso con PyTorch

### Carga del dataset:
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.FashionMNIST('data/', download=True, train=True, transform=transform)
test_set = datasets.FashionMNIST('data/', download=True, train=False, transform=transform)
```

### Normalización
- Los valores de píxeles están en el rango [0, 1]
- Normalización común: `transforms.Normalize((0.5,), (0.5,))` mapea a [-1, 1]
- Alternativa: `transforms.Normalize((0.1307,), (0.3081,))` para normalización estándar

## Ventajas sobre MNIST
- Más desafiante que MNIST (dígitos)
- Mejor representación de problemas de visión por computadora del mundo real
- Mismo formato que MNIST, facilitando la comparación

