# Informe Científico - Versión Resumen (4 páginas)

Esta carpeta contiene una versión condensada del informe científico, optimizada para ocupar aproximadamente 4 páginas.

## Cambios Aplicados

### 1. Teoría (04_teoria.md)
- **Reducido de ~190 líneas a ~25 líneas** (87% de reducción)
- Eliminadas explicaciones detalladas de cada hiperparámetro
- Eliminadas analogías y descripciones extensas
- Mantenida solo la arquitectura base y lista de hiperparámetros evaluados

### 2. Resultados (05_resultados.md)
- **Reducido de ~310 líneas a ~50 líneas** (84% de reducción)
- Eliminados detalles de los 6 experimentos individuales
- Eliminadas tablas de progreso por épocas
- Mantenida solo tabla resumen de mejores resultados
- Eliminadas 6 imágenes de comparación de experimentos
- Mantenidas solo 2 imágenes: curvas finales y matriz de confusión

### 3. Discusión (06_discusion.md)
- **Reducido de ~42 líneas a ~20 líneas** (52% de reducción)
- Eliminado contenido duplicado
- Eliminado análisis del experimento 2 con ADAM
- Condensadas conclusiones a párrafos esenciales

### 4. Introducción (03_introduccion.md)
- **Reducido de ~20 líneas a ~8 líneas** (60% de reducción)
- Eliminadas listas detalladas de características del dataset
- Condensada a información esencial

### 5. Referencias (07_referencias.md)
- **Reducido de 8 referencias a 5 referencias** (37% de reducción)
- Mantenidas solo las referencias esenciales

## Compilación

Para compilar el PDF con formato compacto:

```bash
cd informe_cientifico_resumen
pandoc 01_titulo_autores.md 02_resumen.md 03_introduccion.md 04_teoria.md 05_resultados.md 06_discusion.md 07_referencias.md \
  -o informe_completo.pdf \
  --variable=geometry:margin=1.5cm \
  --variable=fontsize:9pt \
  --variable=linestretch:0.9
```

## Estructura del PDF Resultante

- **Página 1**: Título, autores, resumen, introducción, metodología
- **Página 2**: Resultados (tabla resumen y modelo final)
- **Página 3**: Imágenes (curvas y matriz de confusión) + Discusión
- **Página 4**: Referencias

## Notas

- Los archivos originales en `../informe_cientifico/` no han sido modificados
- Las imágenes se referencian desde `../images/` (directorio padre)
- El formato está optimizado para máximo contenido en mínimo espacio

