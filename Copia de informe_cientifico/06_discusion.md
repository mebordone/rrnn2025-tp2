## Conclusion

Se puede observar una reduccion de coste computacional en relacion al learning rate mediante la variacion del parametro de 0.001 (valor por defecto) a 0.01.

Entrenando con learning rate = 0.001...
Final - Train Acc: 68.48%, Val Acc: 73.47%

Entrenando con learning rate = 0.01...
Final - Train Acc: 83.64%, Val Acc: 83.66%

Con lo que mejora de manera significativa la velocidad de convergencia sin afectar el rendimiento del modelo.

En relacion al Dropout, se puede observar que el valor 0 es el que da los mejores resultados, y ademas se considera que no tiene un impacto significativo su variacion.

Se encontro una arquitectura balanceada en relacion a la cantidad de neuronas de las capas ocultas y la performance del modelo duplicando las neuronas de las dos capas intermedias de 128x64 a 256x128. Si esta cantidad se duplica nuevamente, pasando a 512x256 el resultado no mejora significativamente.


Configuracion optima:
•⁠  ⁠*Optimizador*: ADAM
•⁠  ⁠*Learning Rate*: 0.001
•⁠  ⁠*Arquitectura*: 256-128
•⁠  ⁠*Batch Size*: 32
•⁠  ⁠*Épocas*: 30
•⁠  ⁠*Dropout*: 0.2
•⁠  ⁠*Precisión final en validación*: 87.78%

![Curvas de Entrenamiento Final](images/curvas_entrenamiento_final.png)

La implementacion con el optimizador Adam mejora significativamente la performance. En el modelo con la configuracion optima se puede observar que alcanza un rendimiento maximo en el conjunto de test/validacion a partir de la epoca 10, en el conjunto de train continua mejorando hasta la epoca 25, donde empieza a decaer la precision.

Considerando los resultados obtenidos segun la matriz de confusion, se puede observar que la salida mas problematica para el modelo es la categoria Shirt.

![Matriz de confusion de Entrenamiento Final](images/matriz_confusion_final.png)

Por ultimo, quisieramos mencionar que esta configuracion que es muy eficiente tiene un resultado muy similar al modelo final:

Experimento 2 con ADAM:
*Configuración*: LR=0.001, Batch Size=100, Épocas=10, Dropout=0.2
| Optimizador | Train Accuracy | Validation Accuracy |
| ADAM | 88.56% | 87.38% |

Se considera que la configuracion optima es mucho mas pesada que el experimento 2 con Adam, para ganar solo 0.4% de precision.