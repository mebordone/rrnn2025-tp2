# Informe Científico - Clasificación de Fashion-MNIST con PyTorch

## Estructura del informe

Este directorio contiene el informe científico dividido en secciones independientes para facilitar la edición:

- `01_titulo_autores.md` - Título y datos de autores
- `02_resumen.md` - Resumen/Abstract del trabajo
- `03_introduccion.md` - Introducción y contexto
- `04_teoria.md` - Marco teórico, arquitectura de la red y metodología
- `05_resultados.md` - Resultados y análisis de experimentos
- `06_discusion.md` - Discusión de resultados y conclusiones
- `07_referencias.md` - Referencias bibliográficas

## Cómo compilar el informe completo

Para generar el informe completo, puedes usar uno de estos métodos:

### Método 1: Usando cat (Linux/Mac)
```bash
cat 01_titulo_autores.md 02_resumen.md 03_introduccion.md 04_teoria.md 05_resultados.md 06_discusion.md 07_referencias.md > informe_completo.md
```

### Método 2: Usando pandoc
```bash
pandoc 01_titulo_autores.md 02_resumen.md 03_introduccion.md 04_teoria.md 05_resultados.md 06_discusion.md 07_referencias.md -o informe_completo.pdf
```

### Método 3: Manual
Copia y pega el contenido de cada archivo en orden en un nuevo documento.

## Instrucciones de edición

1. **Completa tus datos personales** en `01_titulo_autores.md`
2. **Ajusta las referencias a figuras** en `05_resultados.md` con los nombres exactos de los archivos en la carpeta `images/`
3. **Personaliza el análisis** en `06_discusion.md` según tus observaciones específicas
4. **Revisa y expande** cualquier sección según sea necesario
5. **Agrega referencias bibliográficas** en `07_referencias.md` según las citas utilizadas en el texto

## Archivos de figuras

Las figuras referenciadas en el informe se encuentran en la carpeta `../images/` y deberían incluir:

### Curvas de entrenamiento:
- Gráficos de pérdida (Cross Entropy Loss) vs épocas para entrenamiento y validación
- Gráficos de precisión (accuracy) vs épocas para entrenamiento y validación

### Análisis de resultados:
- Matriz de confusión (si aplica)
- Ejemplos de imágenes clasificadas correcta e incorrectamente
- Comparación de resultados con diferentes hiperparámetros

### Nota sobre las figuras:
Todas las figuras deben estar referenciadas correctamente en el archivo `05_resultados.md` con descripciones detalladas y numeración secuencial.

## Estructura del trabajo práctico

Este TP2 se enfoca en:
- Clasificación de imágenes del dataset Fashion-MNIST
- Implementación de redes neuronales feedforward multicapa con PyTorch
- Análisis de hiperparámetros (learning rate, optimizador, dropout, etc.)
- Evaluación del rendimiento mediante curvas de entrenamiento/validación

