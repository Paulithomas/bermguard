# Bitácora de Experimentos

Registro de los enfoques evaluados y descartados durante el desarrollo, con sus
resultados numéricos y la causa raíz identificada en cada caso.

Este documento existe porque el proceso de descarte es parte del resultado: las
decisiones adoptadas en el sistema final se entienden mejor a la luz de las
alternativas que se probaron primero.

---

## 1. Segmentación de tomas

**Objetivo.** Determinar los cortes de escena dentro de cada archivo de video,
necesarios para reiniciar tracking y calibración (SUP-09, SUP-19).

**Ground truth.** 16 cortes reales, verificados manualmente reproduciendo los
cuatro videos: 5 en video_01, 3 en video_02, 4 en video_03, 4 en video_04.
Ocurren en múltiplos aproximados de 2 segundos, consistente con concatenación de
clips generativos independientes.

| Detector | Parámetro | Cortes detectados | Resultado |
|---|---|---|---|
| PySceneDetect `ContentDetector` | threshold 27 | 6 (1 falso positivo) | Descartado |
| PySceneDetect `ContentDetector` | threshold 40 | 5 | Descartado |
| PySceneDetect `ContentDetector` | threshold 55 | 4 | Descartado |
| PySceneDetect `AdaptiveDetector` | 3.0 | 6 | Descartado |
| PySceneDetect `AdaptiveDetector` | 1.5 / 1.0 / 0.7 | 12 (saturado) | **Adoptado** |
| ORB, tasa de coincidencia | drop 0.45 | 5 | **Adoptado como complemento** |

**Diagnóstico.** El detector adaptativo satura en 12 cortes: bajar el parámetro
de 1.5 a 0.7 no cambia el resultado. El análisis de la serie temporal de
coincidencia ORB explica por qué: en un corte duro la tasa cae a 0.002, mientras
que en los cortes no detectados solo baja a ~0.47 sobre una mediana de 0.71.

**Causa raíz.** Los 4 cortes no detectados son **transiciones graduales** de 3 a
6 frames, no cortes duros. Ningún método basado en discontinuidad entre frames
consecutivos puede detectarlos.

**Decisión.** Unión de ambos detectores. Se acepta que una toma detectada pueda
contener más de una escena real, siempre de condición lumínica similar, lo que
acota el impacto sobre calibración y preprocesado.

**Hallazgo posterior (§8).** El filtro de plausibilidad de deriva del benchmark
temporal detecta discontinuidades donde estos detectores no ven ninguna, y
sugiere una tercera vía todavía sin explotar.

---

## 2. Estimación del horizonte

**Objetivo.** Localizar la línea de horizonte, requisito de la metrología de
vista única (SUP-14).

| Método | Resultado | Estado |
|---|---|---|
| Punto de fuga de líneas paralelas del suelo | y = 1434 sobre un frame de 1080 px | Descartado |
| Intersección de bases y cimas de dos vehículos | y = 489 ± 4 px, estable entre frames | **Adoptado** |

**Causa del fallo del primer método.** Los caminos del material son curvos. Los
dos bordes de un camino en curva no son paralelos en el mundo real, por lo que su
intersección en la imagen no corresponde al punto de fuga. El resultado cayó
fuera de la imagen, lo que hizo evidente el error.

**Contraste con la estimación automática.** El método adoptado sirvió después
como referencia para validar la calibración del pipeline: sobre video_01, la
estimación automática arroja 478.9 con el método 1 y 485.1 con el método 2,
frente al marcado manual de 489 ± 4. El error es de 2.1% y 0.8% respectivamente.

---

## 3. Clasificación CAEX vs bulldozer

**Objetivo.** Distinguir camión de extracción de maquinaria de oruga. Crítico
porque la calibración usa la altura del vehículo como referencia métrica: 7.4 m
para CAEX contra 4.5 m para dozer.

| Enfoque | Resultado | Estado |
|---|---|---|
| Mapeo de clases COCO | Ambos se detectan como `truck` | Descartado |
| Relación de aspecto del bbox | CAEX de perfil 1.98, dozer 1.94 | Descartado |
| Consistencia física de la altura implicada | Sobre video_03 f90, ambos convergen a h_cámara ≈ 5.4 m | Descartado |
| Prompts de texto explícitos (YOLO-World) | La clase `dozer` se emite, pero no de forma estable | Descartado |

**Causa raíz del tercer método.** Ambigüedad geométrica genuina de la vista
monocular: un vehículo más bajo y más cercano produce exactamente la misma señal
que uno más alto y más lejano. No es un defecto de implementación.

**Precisión sobre el cuarto método.** Con `"bulldozer"` y `"crawler tractor with
blade"` en el vocabulario, sobre video_03 f90 ambos vehículos se etiquetaron
`caex`. En la ejecución completa la clase `dozer` **sí aparece**, pero con 1 a 2
frames de presencia en video_03 y video_04, frente a decenas de frames del mismo
objeto etiquetado `caex`. La clase existe en la salida pero no es estable sobre
un mismo track.

**El fenómeno no es exclusivo del método 2.** En video_02, el método 1 asigna al
ID de seguimiento `#26` la clase `caex` en unos frames y `other_heavy` en otros.
La clasificación no solo es incorrecta: es inconsistente sobre el mismo objeto.

**Conclusión.** Ningún enfoque zero-shot resuelve la clasificación en este
material. Requiere fine-tune sobre datos del dominio.

---

## 4. Extracción de la cresta del pretil

**Objetivo.** Delinear el borde superior del pretil para medir su altura.

| Señal | Resultado | Estado |
|---|---|---|
| Gradiente vertical (Sobel) | Responde a sombras, huellas y rocas antes que a la cresta | Descartado |
| Varianza local + umbral de Otsu | Retiene todo lo texturado, incluidos vehículos | Descartado |
| Varianza local + apertura morfológica horizontal (81×3) | Detecta estructuras horizontales de forma estable | **Adoptado** |

**Por qué falla el gradiente.** La cresta es tierra sobre tierra, de bajo
contraste. Una sombra de camión sobre suelo claro genera mucho más gradiente que
el borde real del camellón.

**Limitación que persiste.** El perfil de energía por fila presenta varios picos
bien formados —talud de fondo, cresta, ripio de primer plano— y el criterio de
máxima energía no siempre selecciona el pretil correcto. El modo de fallo no es
ruido sino **mesetas discretas y sostenidas** en la serie de altura. Afecta a
video_01 y video_02; video_03 y video_04 presentan curvas continuas.

**Idea derivada, explorada después (ver §8).** El pretil es estático en
coordenadas del mundo mientras que huellas y sombras se desplazan con la cámara y
los vehículos. Esa observación, registrada aquí originalmente como trabajo
pendiente, se convirtió en la hipótesis del benchmark de consistencia temporal.

---

## 5. Preprocesado para condiciones nocturnas

**Objetivo.** Recuperar recall en el tramo f46–116 de video_01, donde el método 1
cae a 0.39 detecciones por frame.

| Variante | det/frame | Estado |
|---|---|---|
| Sin preprocesado | 0.39 | Referencia |
| CLAHE clip 2.0 | 0.48 | Descartado, mejora menor |
| CLAHE clip 4.0 | 0.57 | **Recomendado, no integrado** |
| Corrección gamma 1.6 | 0.35 | Descartado, empeora |
| Gamma + CLAHE | 0.39 | Descartado |

**Causa raíz.** Verificación visual del frame 80: hay al menos tres vehículos en
cuadro. Los alejados son siluetas sin textura interna; los cercanos quedan
envueltos en halos de saturación de los faros. **No es un problema de contraste
global sino de ausencia de información local**, y ninguna transformación
fotométrica recupera píxeles saturados.

**Estado real.** CLAHE clip 4.0 es la mejor de las variantes evaluadas y se
recomienda para tomas nocturnas, pero **no está integrado al pipeline**: su
aplicación por toma depende de la segmentación de escena, que tampoco lo está
(SUP-22, SUP-09). La mitigación efectiva que sí opera es el método 2, que triplica
el recall nocturno.

---

## 6. Supuesto refutado: cámara fija

La formulación inicial de SUP-10 asumía cámara estática dentro de cada toma, con
deriva bajo el 2% del ancho. **La medición la refutó**: solo 1 de 6 tomas cumplía
el criterio.

Caracterización posterior mediante homografía ORB+RANSAC sobre ventanas de 10
frames: escala entre 0.984 y 0.996, desplazamiento del centro entre 4 y 19 px,
equivalente a 1–2 px por frame. Es deriva lenta tipo dolly, no cámara fija.

**Consecuencias.** Se descartaron cuatro técnicas que dependían del supuesto:
fondo por mediana temporal, ROI estático, sustracción de fondo MOG2 y calibración
única por toma. Ninguna llegó a implementarse, que es precisamente el beneficio
de haber medido antes de escribir código.

**Nota metodológica.** La primera medición de deriva arrojó valores de hasta 28%
con solo 8–10 inliers por homografía. Se descartó por baja confianza: con tan
pocas correspondencias, RANSAC encuentra alguna transformación que encaja sin que
ello signifique nada. La medición válida se obtuvo sobre ventanas cortas, con 30
inliers mínimos y cobertura declarada. Este criterio se reutilizó después en el
benchmark temporal, donde resultó tener un uso inesperado (§8).

---

## 7. Un error de implementación que vale registrar

La primera versión del medidor de deriva enmascaraba el tercio superior del
frame, buscando evitar que la maquinaria en movimiento contaminara la estimación.
Arrojó **0 inliers en las 6 tomas**.

El diagnóstico fue directo: sobre video_01 f0, el tercio superior rinde 0
keypoints ORB, la banda media 559, y el frame completo 1957. El tercio superior es
cielo sin textura.

Más allá del error, el dato es informativo: **toda la información estructural
explotable del material está concentrada en la banda del terreno**, lo que
refuerza la decisión de SUP-01 de descartar cuanto esté sobre el horizonte. La
corrección se aplicó también al estimador de deriva del benchmark temporal, que
restringe los keypoints a la banda del terreno hacia abajo.

---

## 8. La idea de §4, ya explorada

La observación registrada en §4 —el pretil es estático en el mundo, las sombras y
huellas no— se materializó en un trazador de Viterbi con prior de transición
sobre coordenadas compensadas por deriva.

**Resultado.** Con el parámetro más restrictivo, la desviación absoluta mediana
de la trayectoria de cresta cae entre 25% y 82% según el video, y la tasa de
saltos espurios entre 75% y 95%, a cambio de entre 6% y 25% de la energía
seleccionada.

**Tres hallazgos que no se buscaban:**

1. **La métrica inicial ocultaba el efecto.** La primera versión reportaba
   desviación estándar y no mostraba mejora en ningún video. σ mide la separación
   entre modos, no la estabilidad; con MAD el efecto apareció de inmediato.
2. **La segmentación de tomas domina el resultado.** Video_01, el único con
   cortes bien detectados, alcanza 82% de reducción; video_03, sin cortes
   declarados, se queda en 25%. SUP-09 es el cuello de botella de SUP-01.
3. **El filtro de saneamiento resultó ser un detector de cortes.** Los pasos
   descartados por deriva implausible señalan discontinuidades que los detectores
   de §1 no encuentran: 21 en video_03, que declara cero cortes.

**No está integrado al pipeline.** El detalle completo —método, métricas,
limitaciones y trabajo siguiente— está en `docs/CONSISTENCIA_TEMPORAL.md`, y el
resumen en `docs/reporte_benchmark.md` §4.
