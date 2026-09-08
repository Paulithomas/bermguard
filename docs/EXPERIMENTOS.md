# Bitácora de Experimentos

Registro de los enfoques evaluados y descartados durante el desarrollo, con
sus resultados numéricos y la causa raíz identificada en cada caso.

Este documento existe porque el proceso de descarte es parte del resultado:
las decisiones adoptadas en el sistema final se entienden mejor a la luz de
las alternativas que se probaron primero.

---

## 1. Segmentación de tomas

**Objetivo.** Determinar los cortes de escena dentro de cada archivo de video,
necesarios para reiniciar tracking y calibración (SUP-09, SUP-19).

**Ground truth.** 16 cortes reales, verificados manualmente reproduciendo los
cuatro videos: 5 en video_01, 3 en video_02, 4 en video_03, 4 en video_04.
Ocurren en múltiplos aproximados de 2 segundos, consistente con concatenación
de clips generativos independientes.

| Detector | Parámetro | Cortes detectados | Resultado |
|---|---|---|---|
| PySceneDetect `ContentDetector` | threshold 27 | 6 (1 falso positivo) | Descartado |
| PySceneDetect `ContentDetector` | threshold 40 | 5 | Descartado |
| PySceneDetect `ContentDetector` | threshold 55 | 4 | Descartado |
| PySceneDetect `AdaptiveDetector` | 3.0 | 6 | Descartado |
| PySceneDetect `AdaptiveDetector` | 1.5 / 1.0 / 0.7 | 12 (saturado) | **Adoptado** |
| ORB, tasa de coincidencia | drop 0.45 | 5 | **Adoptado como complemento** |

**Diagnóstico.** El detector adaptativo satura en 12 cortes: bajar el
parámetro de 1.5 a 0.7 no cambia el resultado. El análisis de la serie
temporal de coincidencia ORB explica por qué: en un corte duro la tasa cae a
0.002, mientras que en los cortes no detectados solo baja a ~0.47 sobre una
mediana de 0.71.

**Causa raíz.** Los 4 cortes no detectados son **transiciones graduales** de 3
a 6 frames, no cortes duros. Ningún método basado en discontinuidad entre
frames consecutivos puede detectarlos.

**Decisión.** Unión de ambos detectores. Se acepta que una toma detectada
pueda contener más de una escena real, siempre de condición lumínica similar,
lo que acota el impacto sobre calibración y preprocesado.

---

## 2. Estimación del horizonte

**Objetivo.** Localizar la línea de horizonte, requisito de la metrología de
vista única (SUP-14).

| Método | Resultado | Estado |
|---|---|---|
| Punto de fuga de líneas paralelas del suelo | y = 1434 sobre un frame de 1080 px | Descartado |
| Intersección de bases y cimas de dos vehículos | y = 489 ± 4 px, estable entre frames | **Adoptado** |

**Causa del fallo del primer método.** Los caminos del material son curvos.
Los dos bordes de un camino en curva no son paralelos en el mundo real, por lo
que su intersección en la imagen no corresponde al punto de fuga. El resultado
cayó fuera de la imagen, lo que hizo evidente el error.

---

## 3. Clasificación CAEX vs bulldozer

**Objetivo.** Distinguir camión de extracción de maquinaria de oruga. Crítico
porque la calibración usa la altura del vehículo como referencia métrica: 7.4 m
para CAEX contra 4.5 m para dozer.

| Enfoque | Resultado | Estado |
|---|---|---|
| Mapeo de clases COCO | Ambos se detectan como `truck` | Descartado |
| Relación de aspecto del bbox | CAEX de perfil 1.98, dozer 1.94 | Descartado |
| Consistencia física de la altura implicada | Ambos convergen a h_cámara ≈ 5.4 m | Descartado |
| Prompts de texto explícitos (YOLO-World) | Ambos etiquetados `caex` | Descartado |

**Causa raíz del tercer método.** Ambigüedad geométrica genuina de la vista
monocular: un vehículo más bajo y más cercano produce exactamente la misma
señal que uno más alto y más lejano. No es un defecto de implementación.

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
contraste. Una sombra de camión sobre suelo claro genera mucho más gradiente
que el borde real del camellón.

**Limitación que persiste.** El perfil de energía por fila presenta varios
picos bien formados —talud de fondo, cresta, ripio de primer plano— y el
criterio de máxima energía no siempre selecciona el pretil correcto.

**Ideas no exploradas por restricción de tiempo.** El pretil es estático entre
frames mientras que huellas y sombras se desplazan con la cámara y los
vehículos. Acumular la respuesta de textura sobre varios frames compensados
por movimiento debería hacer emerger la estructura real.

---

## 5. Preprocesado para condiciones nocturnas

**Objetivo.** Recuperar recall en el tramo f46–116 de video_01, donde el
método 1 cae a 0.39 detecciones por frame.

| Variante | det/frame | Estado |
|---|---|---|
| Sin preprocesado | 0.39 | Referencia |
| CLAHE clip 2.0 | 0.48 | — |
| CLAHE clip 4.0 | 0.57 | **Adoptado** |
| Corrección gamma 1.6 | 0.35 | Descartado, empeora |
| Gamma + CLAHE | 0.39 | Descartado |

**Causa raíz.** Verificación visual del frame 80: hay al menos tres vehículos
en cuadro. Los alejados son siluetas sin textura interna; los cercanos quedan
envueltos en halos de saturación de los faros. **No es un problema de contraste
global sino de ausencia de información local**, y ninguna transformación
fotométrica recupera píxeles saturados.

---

## 6. Supuesto refutado: cámara fija

La formulación inicial de SUP-10 asumía cámara estática dentro de cada toma,
con deriva bajo el 2% del ancho. **La medición la refutó**: solo 1 de 6 tomas
cumplía el criterio.

Caracterización posterior mediante homografía ORB+RANSAC sobre ventanas de 10
frames: escala entre 0.984 y 0.996, desplazamiento del centro entre 4 y 19 px,
equivalente a 1–2 px por frame. Es deriva lenta tipo dolly, no cámara fija.

**Consecuencias.** Se descartaron cuatro técnicas que dependían del supuesto:
fondo por mediana temporal, ROI estático, sustracción de fondo MOG2 y
calibración única por toma.

**Nota metodológica.** La primera medición de deriva arrojó valores de hasta
28% con solo 8–10 inliers por homografía. Se descartó por baja confianza: con
tan pocas correspondencias, RANSAC encuentra alguna transformación que encaja
sin que ello signifique nada. La medición válida se obtuvo sobre ventanas
cortas, con 30 inliers mínimos y cobertura declarada.

---

## 7. Un error de implementación que vale registrar

La primera versión del medidor de deriva enmascaraba el tercio superior del
frame, buscando evitar que la maquinaria en movimiento contaminara la
estimación. Arrojó **0 inliers en las 6 tomas**.

El diagnóstico fue directo: sobre video_01 f0, el tercio superior rinde 0
keypoints ORB, la banda media 559, y el frame completo 1957. El tercio superior
es cielo sin textura.

Más allá del error, el dato es informativo: **toda la información estructural
explotable del material está concentrada en la banda del terreno**, lo que
refuerza la decisión de SUP-01 de descartar cuanto esté sobre el horizonte.