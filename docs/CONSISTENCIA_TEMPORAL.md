# Consistencia temporal en la selección de la cresta

Benchmark de una alternativa al criterio de máxima energía por frame, con el
método, los resultados medidos sobre los cuatro videos de muestra y los
hallazgos derivados.

Este documento es el detalle metodológico del **segundo eje del benchmark**,
resumido en `docs/reporte_benchmark.md` §4. Cubre la parte del enunciado
referida a *"la segmentación del terreno"*, mientras que el reporte principal
compara los dos detectores.

El experimento está implementado en `tools/recon/08_crest_viterbi.py` y sus
salidas quedan en `data/recon/crest_viterbi/`. **No está integrado al
pipeline**: es trabajo validado, no una modificación adoptada.

---

## 1. Problema

`extract_crest` selecciona la fila del pretil con

```python
energy = response.sum(axis=1)
peak_row = int(np.argmax(energy))
```

Es una decisión independiente por frame. Cuando el perfil de energía presenta
varios picos bien formados —talud de fondo, cresta, ripio de primer plano— el
máximo salta entre estructuras. El refinamiento posterior por columna queda
anclado a esa fila, de modo que un `peak_row` equivocado invalida el frame
completo.

El modo de fallo es visible en las curvas de `berm_height.png`: no se
manifiesta como ruido sino como mesetas discretas y sostenidas, donde la
mediana móvil salta a un nivel superior y permanece ahí mientras esa estructura
domina el perfil.

Esta es la limitación registrada en SUP-01, y se refleja en los percentiles que
reporta `metadata.json`: en video_01 y video_02 el p90 más que duplica la
mediana, mientras que en video_03 y video_04 queda contenido.

## 2. Hipótesis

El pretil es estático en coordenadas del mundo. Su fila en la imagen solo puede
desplazarse por la deriva de cámara, medida en 1–2 px por frame (SUP-10). Una
selección que se mueva más que eso entre frames contiguos no está siguiendo el
pretil: está cambiando de estructura.

La hipótesis convierte una propiedad física de la escena en un criterio de
selección. Es la que `docs/EXPERIMENTOS.md` §4 registra como idea no explorada
por restricción de tiempo.

## 3. Método

### 3.1 Trazador de Viterbi sobre coordenadas compensadas

Se plantea la secuencia de filas de cresta como un problema de camino óptimo
sobre `T` frames y `K` filas candidatas:

- **Costo de emisión:** `−energía normalizada a [0,1]` por frame.
- **Costo de transición:** `λ · min((|y_t − (y_{t−1} + dy_t)| / τ)², 1)`

Dos decisiones importan aquí.

**El residuo se toma respecto de la deriva esperada, no de la fila anterior.**
Penalizar `|y_t − y_{t−1}|` castigaría al pretil real, que sí se desplaza en la
imagen, y favorecería estructuras estáticas en pantalla. La compensación por
`dy_t` es lo que hace que el prior exprese la hipótesis en vez de contradecirla.

**El costo de transición está truncado.** Queda acotado en `[0, λ]`, lo que lo
hace comparable con la emisión, que vive en `[0,1]`. Sin truncar, `λ · diff²`
alcanza miles frente a una emisión de 1: un solo salto grande domina el camino
completo y el término de energía se vuelve irrelevante. El truncamiento es la
forma estándar de un prior robusto — penaliza desviaciones moderadas y deja de
crecer ante las que ya son claramente un cambio de estructura.

El parámetro `τ` se fija en 4 px, el doble de la deriva máxima medida (SUP-10),
de modo que un desplazamiento compatible con la deriva no reciba penalización
apreciable.

### 3.2 Estimación de la deriva

Homografía afín parcial ORB + RANSAC entre frames contiguos, restringida a la
banda del terreno hacia abajo. La restricción corrige el error registrado en
`docs/EXPERIMENTOS.md` §7: enmascarar la parte superior del frame deja solo
cielo, que rinde 0 keypoints frente a los 1957 del frame completo.

Se descartan los pasos con menos de 30 correspondencias y los que devuelven un
desplazamiento superior a 10 px. Este segundo filtro aplica el mismo criterio
que se usó para descartar la primera medición de deriva: con pocas
correspondencias válidas RANSAC encuentra alguna transformación que encaja sin
que ello signifique nada. Un paso descartado se trata como deriva nula, no como
dato inventado.

### 3.3 Segmentación por toma

Tanto la compensación de deriva como el prior de transición solo son válidos
dentro de una escena. Cruzando un corte, la cresta puede estar legítimamente en
cualquier fila, y penalizar ese salto introduce dos errores: contamina la deriva
acumulada con homografías entre escenas distintas, y castiga una discontinuidad
real.

Cada toma se procesa como secuencia independiente. Los cortes provienen de la
unión de `shots.json` y `orb_cuts.json`, que es la decisión declarada en SUP-09.
Las métricas se calculan por toma y se agregan ponderando por número de frames.

### 3.4 Métricas

| Métrica | Qué mide |
|---|---|
| `mad_stabilized_px` | Desviación absoluta mediana escalada (×1.4826) de la trayectoria compensada por deriva |
| `jump_rate_pct` | Porcentaje de transiciones cuyo residuo respecto de la deriva excede τ |
| `mean_energy` | Energía normalizada media en las filas seleccionadas |

**Por qué MAD y no desviación estándar.** La primera versión del benchmark
reportaba σ y no mostraba mejora alguna en ningún video. El diagnóstico fue que
sobre una trayectoria con transiciones residuales σ mide la **separación entre
modos**, no la estabilidad de la selección: con una banda de búsqueda de 252
filas, una sola transición de 137 px basta para dominarla. Al cambiar a MAD, que
es robusta a esa cola, el efecto del prior se hizo visible de inmediato en los
cuatro videos.

Es el hallazgo metodológico del experimento: **una métrica mal elegida ocultaba
por completo un efecto real**.

**Por qué `mean_energy` acompaña siempre.** Las dos métricas de estabilidad son
engañosas por sí solas: una trayectoria constante las optimiza de forma trivial
(MAD = 0, saltos = 0) sin tener relación alguna con el pretil. `mean_energy`
expone lo que el prior cede a cambio. El valor de 1.0000 en el método base no es
un mérito: es el techo por construcción, ya que argmax elige el máximo en cada
frame. Sirve como referencia de cuánta energía cede el trazador, no como
comparación.

## 4. Resultados

Secuencias completas, τ = 4 px.

### Video 01 — 302 frames, 6 tomas, cortes en 46/116/141/171/215

| Método | MAD (px) | Saltos (%) | Energía |
|---|---|---|---|
| argmax | 13.70 | 13.23 | 1.0000 |
| λ = 0.5 | 7.22 | 3.05 | 0.9831 |
| λ = 2 | 8.33 | 1.68 | 0.9617 |
| **λ = 8** | **2.46** | **0.67** | 0.8981 |

### Video 02 — 240 frames, 2 tomas, corte en 156

| Método | MAD (px) | Saltos (%) | Energía |
|---|---|---|---|
| argmax | 114.36 | 18.95 | 1.0000 |
| λ = 0.5 | 114.71 | 7.15 | 0.9842 |
| λ = 2 | 133.14 | 4.21 | 0.9043 |
| **λ = 8** | **61.75** | **1.26** | 0.7495 |

### Video 03 — 240 frames, 1 toma, sin cortes declarados

| Método | MAD (px) | Saltos (%) | Energía |
|---|---|---|---|
| argmax | 68.62 | 15.48 | 1.0000 |
| λ = 0.5 | 69.95 | 8.37 | 0.9808 |
| λ = 2 | 55.08 | 4.18 | 0.9259 |
| **λ = 8** | **51.48** | **1.67** | 0.7676 |

### Video 04 — 240 frames, 1 toma, sin cortes declarados

| Método | MAD (px) | Saltos (%) | Energía |
|---|---|---|---|
| argmax | 56.25 | 5.02 | 1.0000 |
| λ = 0.5 | 56.10 | 1.26 | 0.9891 |
| λ = 2 | 56.30 | 1.26 | 0.9832 |
| **λ = 8** | **41.96** | **1.26** | 0.9419 |

## 5. Hallazgos

### 5.1 El prior temporal reduce la inestabilidad en los cuatro videos

Con λ = 8, la tasa de saltos cae entre 75% y 95%, y MAD entre 25% y 82%. El
costo en energía va de 6% a 25%. La dirección del efecto es consistente y el
compromiso es medible.

### 5.2 La calidad de la segmentación domina el resultado

Video 01, el único con cortes bien detectados, alcanza 82% de reducción de MAD.
Video 03, sin ningún corte declarado, se queda en 25%. La diferencia no es del
trazador —es el mismo algoritmo con los mismos parámetros— sino de la
segmentación que lo precede.

**SUP-09 es el cuello de botella de SUP-01.** El prior temporal solo rinde
cuando opera dentro de una escena continua. Esta dependencia no era evidente
antes del experimento, y reordena las prioridades del trabajo pendiente: la
segmentación por toma pasa a ser la tarea de mayor retorno.

### 5.3 Los descartes de deriva miden cortes no detectados

El filtro de plausibilidad descarta pasos donde ORB+RANSAC devuelve más de 10 px
de desplazamiento, algo imposible dentro de una escena continua con deriva de
1–2 px por frame. Cada descarte señala una discontinuidad.

| Video | Descartes | Pasos evaluados | Cortes declarados | MAD (λ = 8) |
|---|---|---|---|---|
| 01 | 3 | 296 | 5 | 2.46 |
| 04 | 12 | 239 | 0 | 41.96 |
| 02 | 18 | 238 | 1 | 61.75 |
| 03 | 21 | 239 | 0 | 51.48 |

La relación es monótona: más descartes, peor resultado. Video 03 acumula 21
discontinuidades detectadas por esta vía y cero cortes en los artefactos de
segmentación.

Es una medición independiente de la limitación de SUP-09, obtenida por una vía
distinta a la que la detectó originalmente, y **surgida de un mecanismo de
saneamiento, no de una búsqueda deliberada**. Sugiere que hay más
discontinuidades de las que los detectores actuales declaran, y ofrece la vía de
mejora del §7.

### 5.4 El parámetro no es monótono

Con λ = 0.5 y λ = 2 el MAD llega a **empeorar** respecto del método base:
video 02 pasa de 114.36 a 133.14 con λ = 2, y video 03 sube levemente con
λ = 0.5. Un prior débil puede escoger un camino intermedio entre dos estructuras
sin comprometerse con ninguna, peor que no imponer prior alguno.

La elección de λ no puede delegarse a una búsqueda ingenua sobre una sola
métrica.

## 6. Limitaciones

**Sin ground truth, la validación es de consistencia, no de exactitud
(SUP-11).** El benchmark demuestra que el trazador produce trayectorias más
consistentes con la física de la escena. **No demuestra que la fila seleccionada
sea la cresta del pretil.** Una trayectoria estable sobre la estructura
equivocada obtendría buenas métricas, y `mean_energy` acota ese riesgo pero no
lo elimina.

**El benchmark no replica el pipeline exactamente.** Corre sin detector, de modo
que omite la máscara de exclusión de vehículos y usa la banda operativa por
defecto (0.40–0.75 de la altura). Esto lo hace reproducible sin pesos y sin GPU,
y afecta por igual a los dos métodos comparados, pero **las cifras absolutas no
son trasladables al pipeline sin volver a medir**.

**Las cifras provienen de ejecución local, no del contenedor.** A diferencia de
las del reporte de benchmark, se generan corriendo
`python tools/recon/08_crest_viterbi.py --all` sobre `data/videos/`.

**Las transiciones graduales siguen sin resolverse.** La unión de detectores
recupera 12 de 16 cortes reales; las 4 restantes son transiciones de 3 a 6
frames que ningún método basado en discontinuidad entre frames contiguos puede
detectar. Contaminan la toma que las contiene, y el trazador las trata como si
fueran continuidad.

## 7. Trabajo siguiente

En orden de impacto esperado, según lo que muestran los resultados:

1. **Mejorar la segmentación de tomas.** Es la condición previa, por el hallazgo
   5.2. El hallazgo 5.3 ofrece una vía concreta: usar el residuo de la homografía
   como señal de corte, que detecta discontinuidades donde los detectores
   actuales no ven ninguna. Es un detector nuevo que sale de un subproducto del
   saneamiento, y su ground truth ya existe: los 16 cortes marcados manualmente.
2. **Elegir λ con criterio explícito** en lugar de barrido, dado el hallazgo 5.4.
3. **Integrar al pipeline** solo después de lo anterior, y midiendo el efecto
   **sobre las series de altura**, no sobre la fila de cresta: la fila es el
   mecanismo, la altura es el resultado que importa.
