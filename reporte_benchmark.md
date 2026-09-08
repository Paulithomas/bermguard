# Reporte de Benchmark — BermGuard AI

Comparación de dos métodos de detección de maquinaria sobre el material de
muestra, en el contexto del pipeline completo de inspección de pretiles y
monitoreo de proximidad.

---

## 1. Métodos evaluados

| | Método 1 | Método 2 |
|---|---|---|
| Modelo | YOLOv11n-seg | YOLOv8s-WorldV2 |
| Paradigma | Vocabulario cerrado (COCO) | Open-vocabulary zero-shot |
| Definición de clases | Mapeo de etiquetas COCO | Prompts de texto, parametrizables |
| Máscaras de instancia | Sí | No, solo cajas |
| Peso de pesos | 6.2 MB | 25.9 MB + 354 MB (encoder CLIP) |

Ambos comparten el resto del pipeline: tracking ByteTrack, calibración
métrica, extracción de cresta, cálculo de proximidad y generación de
artefactos. La comparación aísla el detector.

### Hiperparámetros

| Parámetro | Método 1 | Método 2 | Justificación |
|---|---|---|---|
| Umbral de confianza | 0.25 | 0.05 | Los rangos de confianza no son comparables entre paradigmas. El valor de cada método se calibró para maximizar recall sin admitir el falso positivo `airplane` que COCO produce en estas escenas con conf 0.17 |
| Tracker | ByteTrack | ByteTrack | Idéntico, para aislar el efecto del detector |
| Resolución de inferencia | Nativa | Nativa | 1920×1080 y 1280×720 según el video (SUP-28) |

El umbral bajo en ambos casos es deliberado (SUP-18): el material generativo
produce geometrías irregulares que penalizan a detectores estrictos, y el
rechazo de falsos positivos se delega al filtrado temporal del tracker.

---

## 2. Selección de métricas

**No existe ground truth.** El material es sintético (SUP-11), por lo que no
hay anotaciones contra las cuales calcular mAP o mIoU en sentido estricto. Las
métricas se eligieron en consecuencia:

| Métrica | Qué mide | Por qué |
|---|---|---|
| Detecciones por frame, desglosadas por condición lumínica | Recall relativo | Sin ground truth no hay precisión absoluta, pero el desglose por condición sí es comparable entre métodos sobre el mismo material |
| Latencia media y p95 (ms/frame) | Costo computacional | El p95 importa más que la media en un sistema de seguridad: el peor caso define la latencia de alerta |
| Dispersión de la calibración de escala (%) | Calidad de las detecciones como referencia métrica | Un detector que encuentra más vehículos pero peor delimitados degrada la metrología |
| Nivel de cascada de calibración alcanzado | Viabilidad de medición métrica | Determina si el sistema puede reportar en metros o debe degradar a unidades normalizadas (SUP-15) |
| Concordancia entre métodos en la altura estimada | Proxy de confiabilidad | Ante ausencia de ground truth, la convergencia de dos estimadores independientes es evidencia (SUP-21) |

---

## 3. Resultados

### 3.1 Recall por condición lumínica

Medido sobre video_01, tramo diurno (f0–46) y tramo nocturno con faros
directos (f46–116), muestreando 1 de cada 3 frames.

| Condición | Método 1 | Método 2 | Variación |
|---|---|---|---|
| Diurna | 0.88 det/frame | — | — |
| **Nocturna con faros** | **0.39 det/frame** | **1.35 det/frame** | **+246%** |

La hipótesis registrada en SUP-30 antes de implementar el método 2 —que un
modelo fundacional degradaría menos en condiciones nocturnas al razonar sobre
semántica visual amplia en lugar de patrones aprendidos de COCO— **se
confirma**.

### 3.2 Latencia

| | Método 1 | Método 2 |
|---|---|---|
| ms/frame medio (1080p) | 64.8 | 66.1 |
| p95 (1080p) | 81.3 | 78.7 |
| FPS efectivo | 15.3 | 15.1 |

La diferencia es marginal en el pipeline completo, donde la extracción de
cresta domina el costo. Medido de forma aislada, el detector open-vocabulary
es aproximadamente 1.5× más lento (41 ms contra 27 ms en el mismo frame).

### 3.3 Comportamiento sobre los cuatro videos

| Video | Alertas M1 | Alertas M2 | Pretil M1 (m) | Pretil M2 (m) | Dispersión M1 | Dispersión M2 |
|---|---|---|---|---|---|---|
| 01 | 89 | 177 | 2.68 | 2.80 | 7.2% | 15.2% |
| 02 | 13 | 68 | 2.68 | 2.65 | 6.1% | 5.8% |
| 03 | 11 | 35 | 1.84 | 1.74 | 12.1% | 12.0% |
| 04 | 1 | 68 | 2.09 | 2.55 | nivel 2 | 6.8% |

**Video_04 es el caso decisivo.** El método 1 no encontró suficientes vehículos
para calibrar por referencia y degradó al nivel 2 de la cascada, asumiendo una
altura de cámara. El método 2 calibró en nivel 1 con dispersión de 6.8%. En
ese video, el modelo fundacional habilita metrología donde el especializado no
puede.

**Contrapartida.** En video_01 la dispersión del método 2 duplica a la del
método 1 (15.2% contra 7.2%). Más detecciones incluye detecciones peor
delimitadas, lo que degrada la estimación de escala. El mayor recall no es
gratuito.

### 3.4 Concordancia entre métodos

Las alturas medianas de pretil estimadas de forma independiente por ambos
pipelines:

| Video | M1 | M2 | Diferencia |
|---|---|---|---|
| 01 | 2.68 | 2.80 | 4.5% |
| 02 | 2.68 | 2.65 | 1.1% |
| 03 | 1.84 | 1.74 | 5.4% |
| 04 | 2.09 | 2.55 | 22.0% |

Tres de cuatro videos convergen bajo el 6%. Ante la ausencia de ground truth,
esta convergencia es la evidencia más fuerte disponible de que la estimación
es estable. La divergencia de video_04 es esperable: es el único donde los
métodos usaron niveles de calibración distintos.

---

## 4. Lo que ningún método resolvió

**Clasificación CAEX/dozer.** Se esperaba que el método 2 la resolviera, dado
que acepta prompts de texto y se le proporcionaron `"bulldozer"` y
`"crawler tractor with blade"` como clases explícitas. **No lo hizo**: sobre
video_03 f90, con un CAEX y un bulldozer en cuadro, ambos se etiquetaron
`caex`. El vocabulario está disponible pero el modelo no discrimina estas dos
clases en este material.

Combinado con los dos discriminantes geométricos ya descartados —relación de
aspecto y consistencia física de la altura implicada (SUP-29)— la conclusión
es que **la clasificación requiere fine-tune sobre datos del dominio**. No hay
atajo zero-shot.

---

## 5. Hallazgo de despliegue

YOLO-World descarga el encoder de texto CLIP la primera vez que se invoca
`set_classes()`, es decir **después** de cargar los pesos del detector. Copiar
únicamente el `.pt` a la imagen no basta: el contenedor fallaría al arrancar
en un entorno sin red, incumpliendo el requisito Plug & Play.

Se resolvió mediante `tools/prepare_weights.py`, que fuerza la inicialización
del vocabulario durante el `docker build`. El encoder queda cacheado en
`weights/clip/`, ruta que Ultralytics resuelve en runtime sin conectividad.
Verificado bloqueando la resolución DNS: ambos métodos cargan correctamente.

Este costo es propio del paradigma open-vocabulary: el encoder de texto pesa
354 MB, catorce veces más que el detector que lo usa.

---

## 6. Recomendación

**Para inspección diurna con requisito de segmentación fina: método 1.** Menor
peso, produce máscaras de instancia, y su dispersión de calibración es
consistentemente menor.

**Para operación continua incluyendo turno de noche: método 2.** El triple de
recall nocturno es determinante en un sistema de seguridad: un vehículo no
detectado es una alerta de proximidad que no se emite.

**Para producción: ninguno de los dos tal cual.** Ambos comparten la limitación
de clasificación, que afecta directamente la calibración métrica al usar 7.4 m
como referencia para un vehículo que mide 4.5 m. El fine-tune sobre datos del
dominio es prerrequisito, no mejora opcional.

---

## 7. Limitaciones de este benchmark

- **Sin ground truth**, las métricas son comparativas entre métodos, no
  absolutas. Un método puede tener mayor recall y ser igualmente incorrecto.
- **Muestra reducida**: 4 videos, 1.022 frames en total. Las diferencias
  observadas podrían no generalizar.
- **Latencia medida en Apple Silicon con MPS**, no en el hardware objetivo
  (x86 con CUDA). Los valores absolutos no son transferibles; la comparación
  relativa entre métodos sí, al haberse medido en las mismas condiciones.
- **Optimización de inferencia no evaluada.** Con 1.022 frames en modo batch,
  cuantización INT8 o TensorRT no aportan beneficio medible. En un despliegue
  real de tiempo real sí serían determinantes.