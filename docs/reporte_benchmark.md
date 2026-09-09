# Reporte de Benchmark — BermGuard AI

Comparación de dos métodos de detección de maquinaria sobre el material de
muestra, en el contexto del pipeline completo de inspección de pretiles y
monitoreo de proximidad.

**Procedencia de las cifras.** Salvo donde se indique lo contrario, todos los
valores provienen de una única ejecución de `--method all` sobre runner Linux
x86_64 **sin GPU** (GitHub Actions), a partir del estado versionado del
repositorio. Los ocho `metadata.json` que las respaldan se regeneran ejecutando
el contenedor sobre `data/videos/`.

---

## 1. Métodos evaluados

| | Método 1 | Método 2 |
|---|---|---|
| Modelo | YOLOv11n-seg | YOLOv8s-WorldV2 |
| Paradigma | Vocabulario cerrado (COCO) | Open-vocabulary zero-shot |
| Definición de clases | Mapeo de etiquetas COCO | Prompts de texto, parametrizables |
| Máscaras de instancia | Sí | No, solo cajas |
| Peso de pesos | 6.2 MB | 25.9 MB + 354 MB (encoder CLIP) |

Ambos comparten el resto del pipeline: tracking ByteTrack, calibración métrica,
extracción de cresta, cálculo de proximidad y generación de artefactos. La
comparación aísla el detector.

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

**No existe ground truth.** El material es sintético (SUP-11), por lo que no hay
anotaciones contra las cuales calcular mAP o mIoU en sentido estricto. Las
métricas se eligieron en consecuencia:

| Métrica | Qué mide | Por qué |
|---|---|---|
| Detecciones por frame, desglosadas por condición lumínica | Recall relativo | Sin ground truth no hay precisión absoluta, pero el desglose por condición sí es comparable entre métodos sobre el mismo material |
| Latencia media y p95 (ms/frame) | Costo computacional | El p95 importa más que la media en un sistema de seguridad: el peor caso define la latencia de alerta |
| Dispersión de la calibración de escala (%) | Calidad de las detecciones como referencia métrica | Un detector que encuentra más vehículos pero peor delimitados degrada la metrología |
| Nivel de cascada de calibración alcanzado | Viabilidad de medición métrica | Determina si el sistema puede reportar en metros o debe degradar a unidades normalizadas (SUP-15) |
| Cobertura del pretil (% de frames medibles) | Efecto del tipo de salida del detector sobre la etapa siguiente | Máscara y caja excluyen maquinaria con precisión distinta, y eso afecta al extractor de cresta |
| Distancia mínima entre pares concurrentes | Consistencia de la identidad de los tracks | Un valor físicamente imposible delata fragmentación de IDs, medible con un umbral físico |
| Concordancia entre métodos en la altura estimada | Proxy de confiabilidad | Ante ausencia de ground truth, la convergencia de dos estimadores independientes es evidencia (SUP-21) |

---

## 3. Resultados

### 3.1 Recall por condición lumínica

> **Procedencia distinta.** Estas cifras provienen de una medición dirigida
> sobre tramos de video_01, muestreando 1 de cada 3 frames, y no de la
> ejecución completa del pipeline. Se conservan porque aíslan el efecto que
> ninguna métrica agregada captura, pero no son reproducibles ejecutando el
> contenedor.

| Condición | Método 1 | Método 2 | Variación |
|---|---|---|---|
| Diurna (f0–46) | 0.88 det/frame | — | — |
| **Nocturna con faros (f46–116)** | **0.39 det/frame** | **1.35 det/frame** | **+246%** |

La hipótesis registrada en SUP-30 antes de implementar el método 2 —que un
modelo fundacional degradaría menos en condiciones nocturnas al razonar sobre
semántica visual amplia en lugar de patrones aprendidos de COCO— **se
confirma**.

El efecto es coherente con el conteo de alertas de 3.3, donde el método 2
multiplica las detecciones en los cuatro videos, aunque ese agregado mezcla
recall real con fragmentación de tracks.

---

### 3.2 Latencia

Ejecución en CPU sobre las secuencias completas.

| Video | Resolución | ms/frame M1 | ms/frame M2 | p95 M1 | p95 M2 | FPS M1 | FPS M2 |
|---|---|---|---|---|---|---|---|
| 01 | 1920×1080 | 159.2 | 230.6 | 190.9 | 264.7 | 6.28 | 4.34 |
| 02 | 1280×720 | 115.4 | 180.0 | 144.8 | 210.6 | 8.66 | 5.55 |
| 03 | 1280×720 | 117.7 | 188.5 | 144.4 | 215.4 | 8.49 | 5.31 |
| 04 | 1280×720 | 119.0 | 183.2 | 142.1 | 213.8 | 8.41 | 5.46 |

**El método 2 es entre 45% y 56% más lento**, de forma consistente en los cuatro
videos. La penalización no es marginal ni depende de la resolución: es el costo
estructural de un detector open-vocabulary, que evalúa el vocabulario de texto
además de la imagen.

El p95 se reporta junto a la media porque en un sistema de seguridad el peor
caso define la latencia de alerta. La razón p95/media es estable (~1.18 en ambos
métodos), lo que indica baja variabilidad por frame y ausencia de picos
patológicos en cualquiera de los dos.

**Contexto de la medición.** Estos valores son de CPU. En el hardware objetivo
(x86_64 con CUDA) los absolutos serán sustancialmente menores. La comparación
relativa sí es trasladable, al haberse medido en las mismas condiciones y en la
misma ejecución.

**Variabilidad entre ejecuciones.** Dos ejecuciones del mismo commit sobre
runners de CI arrojaron FPS distintos (video_01/M1: 6.28 y 4.15; video_01/M2:
4.34 y 2.55), por carga variable de CPU compartida. **Los valores absolutos de
latencia no son reproducibles en este entorno; la razón entre métodos sí**
(1.45 y 1.63). Todas las demás métricas —altura, percentiles, dispersión,
alertas, cobertura, distancia mínima, horizonte— son idénticas entre ambas
ejecuciones: el pipeline es determinista salvo en la medición de tiempo.

---

### 3.3 Comportamiento sobre los cuatro videos

| Video | Alertas M1 | Alertas M2 | Pretil M1 (m) | Pretil M2 (m) | Dispersión M1 | Dispersión M2 |
|---|---|---|---|---|---|---|
| 01 | 86 | 193 | 2.75 | 3.14 | 6.9% | 15.2% |
| 02 | 15 | 71 | 2.69 | 2.54 | 5.9% | 6.3% |
| 03 | 13 | 34 | 1.86 | 1.69 | 11.8% | 15.4% |
| 04 | 4 | 67 | 2.14 | 2.55 | nivel 2 | 6.9% |

**Video_04 es el caso decisivo.** El método 1 no reunió referencia vehicular
suficiente y degradó al nivel 2 de la cascada (`assumed_camera_height`,
confianza 0.40, sin dispersión calculable). El método 2 calibró en nivel 1 con
dispersión de 6.9% y confianza 0.86. **En ese video el modelo fundacional
habilita metrología donde el especializado no puede.**

**Contrapartida en la calidad de la referencia.** En video_01 la dispersión del
método 2 duplica a la del método 1 (15.2% contra 6.9%), y en video_03 también la
supera (15.4% contra 11.8%). Más detecciones incluye detecciones peor
delimitadas, y la calibración por altura aparente de bounding box es sensible a
esa delimitación. **El mayor recall no es gratuito.** Por la regla de decisión de
SUP-17, ambos casos del método 2 caen en la banda 10–25%, que exige banda de
incertidumbre obligatoria en el reporte.

En video_02, donde ambos calibran en nivel 1 sobre material bien iluminado, la
diferencia de dispersión es despreciable (5.9% contra 6.3%). La ventaja del
método 1 en este eje aparece cuando las condiciones se degradan, no de forma
uniforme.

#### Cobertura del pretil

Fracción de frames con altura de pretil medible.

| Video | Cobertura M1 | Cobertura M2 |
|---|---|---|
| 01 | 92.4% | 81.1% |
| 02 | 100% | 85.4% |
| 03 | 96.3% | 92.1% |
| 04 | 100% | 87.9% |

**El método 1 domina en los cuatro videos**, con ventaja de 4 a 15 puntos. La
causa es arquitectónica: `extract_crest` excluye los píxeles de maquinaria antes
de calcular el perfil de energía, y el método 1 aporta máscaras de instancia
mientras que el método 2 solo aporta cajas. Una caja excluye una región
rectangular que incluye terreno válido alrededor del vehículo; una máscara
excluye únicamente el vehículo.

Es una ventaja del método 1 que no se deriva de su recall sino **del tipo de
salida que produce**, y no aparece en ninguna otra métrica.

#### Distancia mínima entre equipos

Distancia mínima registrada entre pares de equipos concurrentes (SUP-31).

| Video | M1 | Par | M2 | Par |
|---|---|---|---|---|
| 01 | 1.49 m | `caex #47` / `caex #97` | 5.89 m | `caex #92` / `caex #91` |
| 02 | **0.08 m** | `caex #26` / `other_heavy #27` | — | sin pares concurrentes |
| 03 | 5.53 m | `caex #21` / `caex #19` | 10.58 m | `caex #2` / `caex #8` |
| 04 | 2.45 m | `caex #6` / `caex #7` | 17.91 m | `caex #7` / `caex #9` |

**Los valores del método 1 en video_01, video_02 y video_04 son físicamente
imposibles.** Un CAEX mide del orden de 9 m de ancho (SUP-07); dos de ellos con
puntos de contacto a 0.08 m, 1.49 m o 2.45 m ocuparían el mismo espacio.

El caso de video_02 es el más informativo: 8 cm de separación entre un track
etiquetado `caex` y otro etiquetado `other_heavy`. **Es el mismo objeto detectado
dos veces**, con dos IDs y dos clases simultáneas.

Esto convierte una limitación hasta ahora declarada cualitativamente —la
detección intermitente genera IDs nuevos al reaparecer un vehículo— en una
medición con **umbral físico**: si dos tracks están a menos del ancho nominal del
equipo, no son dos equipos. El método 2 no presenta el fenómeno en ninguno de los
tres videos donde produce pares, lo que sugiere que su mayor continuidad de
detección reduce la fragmentación de tracks.

**Consecuencia sobre el conteo de alertas.** Las cifras de `proximity_alerts` del
método 1 están infladas por estos pares espurios: un vehículo duplicado genera
alerta crítica permanente contra sí mismo. Las comparaciones de alertas entre
métodos deben leerse con esa salvedad, y el conteo absoluto no debe usarse como
métrica de calidad de detección.

---

### 3.4 Concordancia entre métodos

Alturas medianas de pretil estimadas de forma independiente por ambos pipelines.

| Video | M1 (m) | M2 (m) | Diferencia |
|---|---|---|---|
| 02 | 2.69 | 2.54 | 5.6% |
| 03 | 1.86 | 1.69 | 9.1% |
| 01 | 2.75 | 3.14 | 14.2% |
| 04 | 2.14 | 2.55 | 19.2% |

**Dos de los cuatro videos concuerdan bajo el 10%; ninguno baja del 5%.** La
convergencia es más débil de lo deseable para sostener SUP-21 como sustituto del
ground truth ausente.

**Interpretación.** La divergencia no es aleatoria y se explica por dos causas
identificables:

- **Video_04 (19.2%)** es el único donde ambos métodos usaron **niveles distintos
  de la cascada de calibración**. Comparar una medición de nivel 1 contra una de
  nivel 2 no es comparar dos estimadores del mismo objeto.
- **Video_01 (14.2%)** es uno de los dos videos afectados por el modo de fallo de
  SUP-01, donde el criterio de máxima energía engancha estructuras distintas.
  Ambos métodos miden, pero no necesariamente la misma estructura.

Descontando esos dos casos, los videos donde ambos calibran al mismo nivel y el
extractor de cresta se comporta de forma estable concuerdan en 5.6% y 9.1%.

**Qué conserva y qué pierde SUP-21.** El mecanismo sigue siendo válido: dos
estimadores independientes sobre el mismo material producen información legítima
sin necesidad de etiquetas. Lo que estas cifras muestran es que **la concordancia
mide el acuerdo del sistema completo, no solo del detector**. Cuando la cascada
de calibración o el extractor de cresta divergen, la concordancia lo refleja. Eso
lo hace un indicador útil de salud del pipeline, pero un sustituto débil del
ground truth.

#### Estimación del horizonte

Eje adicional de concordancia, contrastable contra el marcado manual disponible
(recon 07).

| Video | M1 | M2 | Manual | Error M1 | Error M2 |
|---|---|---|---|---|---|
| 01 | 478.9 | 485.1 | 489 ± 4 | 2.1% | 0.8% |
| 03 | 325.2 | 372.5 | 395.8 | 17.8% | 5.9% |

**El método 2 estima mejor el horizonte en ambos casos.** Es coherente con su
mayor recall: el horizonte se obtiene por intersección de bases y cimas de
vehículos a distinta profundidad (SUP-14), de modo que más vehículos disponibles
produce una estimación mejor condicionada.

Esto matiza el compromiso descrito arriba. El método 2 estima **mejor el
horizonte** y **peor la escala**: aporta más puntos de apoyo geométrico, pero cada
uno peor delimitado. Son dos efectos opuestos del mismo mayor recall, y afectan a
etapas distintas de la calibración.

---

## 4. Segundo eje del benchmark: selección de la cresta

El enunciado pide evaluar métodos para *"la segmentación del terreno y
detección"*. Las secciones anteriores cubren el detector; este eje cubre la
extracción del pretil.

El extractor por textura direccional selecciona la fila de cresta con el criterio
de máxima energía **por frame**, una decisión independiente entre frames. Se
evaluó como alternativa un **trazador de Viterbi** con prior de transición sobre
coordenadas compensadas por deriva de cámara, apoyado en que el pretil es
estático en coordenadas del mundo y solo puede desplazarse en la imagen a razón
de la deriva medida (1–2 px/frame, SUP-10).

Resultados sobre las secuencias completas, con segmentación por toma:

| Video | MAD argmax | MAD Viterbi (λ=8) | Reducción | Saltos argmax | Saltos Viterbi | Energía cedida |
|---|---|---|---|---|---|---|
| 01 | 13.70 px | 2.46 px | 82% | 13.23% | 0.67% | 10% |
| 02 | 114.36 px | 61.75 px | 46% | 18.95% | 1.26% | 25% |
| 03 | 68.62 px | 51.48 px | 25% | 15.48% | 1.67% | 23% |
| 04 | 56.25 px | 41.96 px | 25% | 5.02% | 1.26% | 6% |

**La calidad de la segmentación de tomas domina el resultado.** Video_01, el único
con cortes bien detectados, alcanza 82% de reducción; video_03, sin cortes
declarados, se queda en 25%. La diferencia no es del trazador —es el mismo
algoritmo con los mismos parámetros— sino de la segmentación que lo precede.

**No está integrado al pipeline.** Es trabajo validado, no una modificación
adoptada. El método, las métricas y las limitaciones están en
`docs/CONSISTENCIA_TEMPORAL.md`.

---

## 5. Lo que ningún método resolvió

**Clasificación CAEX/dozer.** Se esperaba que el método 2 la resolviera, dado que
acepta prompts de texto y se le proporcionaron `"bulldozer"` y `"crawler tractor
with blade"` como clases explícitas. **No lo hizo**: sobre video_03 f90, con un
CAEX y un bulldozer en cuadro, ambos se etiquetaron `caex`. El vocabulario está
disponible pero el modelo no discrimina estas dos clases en este material.

Combinado con los dos discriminantes geométricos ya descartados —relación de
aspecto (1.98 contra 1.94) y consistencia física de la altura implicada, esta
última una ambigüedad geométrica genuina de la vista monocular (SUP-29)— la
conclusión es que **la clasificación requiere fine-tune sobre datos del dominio**.
No hay atajo zero-shot.

**Recall nocturno.** El método 2 lo mejora en 246% pero no lo resuelve. La causa
es saturación del sensor por faros directos: los vehículos cercanos quedan
envueltos en halos y los lejanos son siluetas sin textura interna. **No hay
información local que recuperar**, y ninguna transformación fotométrica la
reconstruye (SUP-30).

**Estabilidad de identidad de los tracks.** Ambos métodos fragmentan IDs, el
método 1 de forma medible y severa (§3.3). Ninguno de los dos incorpora
reidentificación por apariencia.

---

## 6. Hallazgo de despliegue

YOLO-World descarga el encoder de texto CLIP la primera vez que se invoca
`set_classes()`, es decir **después** de cargar los pesos del detector. Copiar
únicamente el `.pt` a la imagen no basta: el contenedor fallaría al arrancar en
un entorno sin red, incumpliendo el requisito Plug & Play.

Se resolvió mediante `tools/prepare_weights.py`, que fuerza la inicialización del
vocabulario durante el `docker build`. El encoder queda cacheado bajo
`/app/weights`, ruta que Ultralytics resuelve en runtime sin conectividad.

Este costo es propio del paradigma open-vocabulary: **el encoder de texto pesa 354
MB, catorce veces más que el detector que lo usa**. Es un factor decisivo en
despliegues con restricción de tamaño de imagen que ninguna métrica de precisión
o latencia refleja.

---

## 7. Recomendación

**Para inspección con requisito de segmentación fina: método 1.** Menor peso (6.2
MB contra 380 MB), 45–56% más rápido, produce máscaras de instancia, y alcanza
mayor cobertura del pretil en los cuatro videos. Su dispersión de calibración es
mejor cuando ambos métodos calibran al mismo nivel y las condiciones no se
degradan.

**Para operación continua incluyendo turno de noche: método 2.** El triple de
recall nocturno es determinante en un sistema de seguridad: un vehículo no
detectado es una alerta de proximidad que no se emite. Además es el único que
calibra en nivel 1 sobre los cuatro videos, y estima mejor el horizonte.

**Para producción: ninguno de los dos tal cual.** Tres razones, en orden de
gravedad:

1. **La clasificación CAEX/dozer no está resuelta**, y afecta directamente la
   calibración métrica al usar 7.4 m como referencia para un vehículo que mide
   4.5 m. El fine-tune sobre datos del dominio es prerrequisito, no mejora
   opcional.
2. **La fragmentación de IDs infla el conteo de alertas** y produce distancias
   físicamente imposibles. Requiere reidentificación por apariencia.
3. **El pipeline no segmenta por toma**, lo que degrada calibración, tracking y
   filtrado temporal al atravesar cortes de escena (SUP-09, SUP-19, SUP-22).

---

## 8. Limitaciones de este benchmark

- **Sin ground truth**, las métricas son comparativas entre métodos, no absolutas.
  Un método puede tener mayor recall y ser igualmente incorrecto.
- **Muestra reducida**: 4 videos, 1.022 frames en total. Las diferencias
  observadas podrían no generalizar al set ciego.
- **El conteo de alertas no es una métrica limpia de calidad de detección**, por
  la fragmentación de IDs cuantificada en §3.3. Se reporta por transparencia, no
  como criterio de selección.
- **La latencia se midió en CPU**, no en el hardware objetivo con CUDA. Los
  valores absolutos no son transferibles; la comparación relativa sí, al haberse
  medido en la misma ejecución.
- **Las cifras de recall por condición lumínica (§3.1) provienen de una medición
  dirigida separada** y no se regeneran ejecutando el contenedor.
- **Optimización de inferencia no evaluada.** Con 1.022 frames en modo batch,
  cuantización INT8 o TensorRT no aportan beneficio medible. En un despliegue real
  de tiempo real sí serían determinantes, y la brecha entre 5–8 FPS en CPU y el
  requisito de tiempo real es precisamente donde esas técnicas se justificarían.
- **El segundo eje del benchmark (§4) no está integrado al pipeline.** Sus cifras
  provienen de `tools/recon/08_crest_viterbi.py`, no de la ejecución del
  contenedor.
