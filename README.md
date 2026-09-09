# BermGuard AI

Pipeline de visión artificial para inspección de pretiles de seguridad y
monitoreo de proximidad de maquinaria en botaderos mineros.

---

## Ejecución

```bash
docker build -t bermguard:latest .

docker run --rm --gpus all \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1
```

El entrypoint acepta tanto la invocación anterior como la forma idiomática de
Docker (`bermguard:latest --input ...`) y la ejecución sin argumentos, que usa
las rutas por defecto `/app/test` y `/app/output`.

**En equipos sin GPU NVIDIA, omitir `--gpus all`.** El dispositivo se resuelve en
cascada `cuda → mps → cpu` (SUP-23). La imagen se construye con
`--platform linux/amd64`, por lo que en Apple Silicon corre bajo emulación.

`--method` acepta `1`, `2` o `all`. Con `all` se generan subdirectorios
`method_1/` y `method_2/` dentro de la carpeta de cada video.

### Ejecución reproducible en CI

El repositorio incluye `.github/workflows/run.yml`, que construye la imagen y
ejecuta el pipeline sobre `data/videos/` en un runner Linux x86_64 limpio. Se
lanza manualmente desde la pestaña Actions y publica los artefactos como
descarga. Todas las cifras de este README y del reporte de benchmark provienen
de esa ejecución.

Los cuatro videos de prueba están versionados en `data/videos/`.

---

## Artefactos generados

Por cada video de entrada se crea un subdirectorio con:

| Archivo | Contenido |
|---|---|
| `<video>_osd.mp4` | Video con detecciones, semáforo de proximidad, delineación de pretil y panel de métricas |
| `berm_height.png` | Curva temporal de altura del pretil, cruda y suavizada, con tramos sin dato marcados |
| `vehicle_distribution.png` | Mapa de posiciones por nivel de riesgo, matriz de distancias mínimas entre equipos y permanencia por equipo |
| `metadata.json` | Métricas de inferencia, estadísticos de altura, proximidad mínima y trazabilidad de la calibración |

Los cuatro artefactos se escriben siempre, incluso ante detección nula. Un
resultado nulo bien documentado es un resultado; un directorio vacío es un fallo
de ingeniería.

---

## Documentación

| Documento | Contenido |
|---|---|
| **[docs/SUPUESTOS.md](docs/SUPUESTOS.md)** | Registro de decisiones ante las ambigüedades del enunciado, con IDs estables `SUP-NN`. Incluye el estado de implementación real de cada supuesto |
| **[docs/reporte_benchmark.md](docs/reporte_benchmark.md)** | Comparación de los dos métodos: métricas, hiperparámetros, resultados y recomendación |
| **[docs/CONSISTENCIA_TEMPORAL.md](docs/CONSISTENCIA_TEMPORAL.md)** | Benchmark de una alternativa al criterio de selección de cresta, con método y hallazgos |
| **[docs/EXPERIMENTOS.md](docs/EXPERIMENTOS.md)** | Bitácora de enfoques evaluados y descartados, con la causa raíz de cada fallo |

El registro de supuestos distingue explícitamente entre lo **medido**, lo
**adoptado** y lo **implementado**. Su §7 consolida qué decisiones de diseño están
materializadas en el código y cuáles no.

---

## Arquitectura

```
main.py                      CLI y validación de argumentos
src/
├── detection/               base.py define el contrato que implementan los métodos
│   ├── yolo_seg.py          método 1 — YOLOv11n-seg, vocabulario cerrado
│   ├── yolo_world.py        método 2 — YOLOv8s-WorldV2, open-vocabulary
│   └── factory.py           construcción por nombre de método
├── berm/                    extracción de cresta y perfil de altura
├── geometry/                calibración métrica y cálculo de proximidad
├── osd/                     renderizado del On-Screen Display
├── analytics/               generación de gráficos
├── utils/                   resolución de dispositivo de cómputo
└── pipeline.py              orquestación
configs/default.yaml         umbrales y parámetros, sin recompilar
tools/
├── prepare_weights.py       descarga de pesos en build-time
└── recon/                   scripts de reconocimiento y benchmark
docs/                        supuestos, benchmark, experimentos
```

`VehicleDetector` es una clase abstracta: agregar un método nuevo significa
implementar esa interfaz y registrarlo en la fábrica, sin tocar el orquestador.

---

## Decisiones de ingeniería

Las decisiones adoptadas ante las ambigüedades del enunciado están registradas
en **[docs/SUPUESTOS.md](docs/SUPUESTOS.md)**, con identificadores estables
`SUP-NN` referenciados desde el código, la configuración y este documento.

Las más relevantes:

### Escala métrica sin calibración de cámara (SUP-15)

No hay parámetros intrínsecos disponibles. La escala se resuelve por metrología
de vista única usando la maquinaria como referencia de altura conocida (7.4 m,
CAEX clase 300 t, SUP-07), mediante una cascada de tres niveles que declara
siempre su origen y su confianza en `metadata.json`.

Sin ground truth físico (SUP-11), la validación disponible es de **consistencia,
no de exactitud**. Tres evidencias la sustentan:

**Dispersión intra-video.** La estimación de escala se repite sobre frames
muestreados y se reporta su dispersión: 5.9% en video_02, 6.9% en video_01 y
11.8% en video_03. En video_04 la cascada no encontró referencia vehicular
suficiente y degradó al nivel 2 (`assumed_camera_height`, confianza 0.40),
declarándolo explícitamente en lugar de emitir un número sin respaldo.

**Horizonte contra marcado manual.** La estimación automática de video_01
(y = 478.9) coincide dentro del 2.1% con el marcado manual por intersección de
bases y cimas de dos vehículos (y = 489 ± 4), documentado en
`docs/EXPERIMENTOS.md` §2.

**Convergencia entre métodos.** Dos de los cuatro videos arrojan alturas de
pretil que difieren menos del 10% entre el método 1 y el método 2, que usan
detectores independientes. Los dos que divergen tienen causa identificada:
video_04 es el único donde ambos métodos usaron niveles distintos de la cascada,
y video_01 está afectado por el modo de fallo de SUP-01. La convergencia mide el
acuerdo del sistema completo, no solo del detector, lo que la vuelve un
indicador útil de salud del pipeline pero un sustituto débil del ground truth.

### El material es sintético y carece de ground truth (SUP-11)

No existe verdad física contra la cual validar. El objetivo declarado es
trazabilidad y consistencia, no exactitud absoluta. La dispersión de la
estimación de escala se mide y se reporta en cada `metadata.json`.

La medición mostró que la consistencia proyectiva del material es mejor de lo
previsto (2.5% de dispersión intra-escena), lo que traslada la responsabilidad
del error a los supuestos del sistema y no al generador.

### La cámara no es fija (SUP-10)

El supuesto inicial fue **refutado empíricamente**: solo 1 de 6 tomas cumplía el
criterio de deriva bajo el 2%. Se midió deriva continua de 1–2 px por frame, tipo
dolly. Se descartaron en consecuencia cuatro técnicas que dependían del supuesto:
fondo por mediana temporal, ROI estático, sustracción de fondo MOG2 y calibración
única por toma.

---

## Limitaciones conocidas

### Clasificación CAEX/dozer (SUP-29)

COCO no contiene clase para maquinaria de oruga; ambos vehículos se detectan como
`truck`. Se evaluaron y descartaron **cuatro** discriminantes sin datos
etiquetados:

1. Mapeo directo de clases COCO — ambos son `truck`.
2. Relación de aspecto — un CAEX de perfil alcanza 1.98, indistinguible del 1.94
   de un dozer.
3. Consistencia física de la altura implicada — ambigüedad geométrica genuina de
   la vista monocular: un vehículo más bajo y más cercano produce la misma señal
   que uno más alto y más lejano.
4. Prompts de texto explícitos en el método 2 — ambos se etiquetan `caex` pese a
   tener `"bulldozer"` en el vocabulario.

Se conserva la clase única en la línea base. **Ningún enfoque zero-shot resuelve
la clasificación en este material.**

### Selección de la cresta del pretil (SUP-01)

El extractor detecta estructuras horizontales de forma estable, pero el criterio
de máxima energía no siempre selecciona el pretil correcto. El modo de fallo es
identificable en las curvas de `berm_height.png`: no se manifiesta como ruido,
sino como **mesetas discretas y sostenidas** —la mediana móvil salta a un nivel
superior y permanece ahí mientras esa estructura domina el perfil de energía.
Afecta a video_01 y video_02; video_03 y video_04 presentan curvas continuas.

Se evaluó una alternativa —trazador de Viterbi con prior de transición sobre
coordenadas compensadas por deriva— que reduce la desviación absoluta mediana
entre 25% y 82% según el video. **No está integrada al pipeline.** Método y
hallazgos en [docs/CONSISTENCIA_TEMPORAL.md](docs/CONSISTENCIA_TEMPORAL.md).

### Recall nocturno (SUP-30)

El método 1 cae de 0.88 a 0.39 detecciones por frame en escenas nocturnas con
faros directos. Se verificó visualmente la presencia de al menos tres vehículos
en cuadro frente a 0.39 detectados. La causa es **ausencia de información local
por saturación**, no falta de contraste global.

El preprocesado CLAHE clip 4.0 recupera un 45% en la medición dirigida, pero
**no está integrado al pipeline**: su aplicación depende de la segmentación por
toma, que tampoco lo está. La mitigación efectiva es el método 2, que triplica el
recall nocturno.

### Segmentación de tomas (SUP-09)

Se detectan 12 de 16 cortes reales. Los no detectados son transiciones graduales
de 3 a 6 frames, y ningún método basado en discontinuidad entre frames
consecutivos puede captarlos.

**La detección de tomas opera solo en las herramientas de reconocimiento.** El
pipeline de inferencia procesa cada archivo como unidad única: calibración,
tracking y filtrado temporal no se segmentan por toma. Es el prerrequisito
pendiente de mayor impacto, porque desbloquea también SUP-19 y SUP-22.

### Fragmentación de IDs de tracking

La detección intermitente genera IDs nuevos al reaparecer un vehículo. La matriz
de distancias mínimas lo hace medible: en video_02, el método 1 registra 0.08 m
entre dos tracks de clases distintas, físicamente imposible entre dos equipos de
9 m de ancho. Es el mismo objeto contado dos veces, e infla el conteo de alertas
de proximidad.

---

## Trabajo pendiente

En orden de impacto, según lo que muestran las mediciones:

1. **Segmentación por toma dentro del pipeline (SUP-09).** Una sola tarea que
   desbloquea tres supuestos: reinicialización de tracking en cortes (SUP-19),
   preprocesado adaptativo (SUP-22) y con él la aplicación del CLAHE de SUP-30.
   Además es la condición previa para que el trazador temporal de cresta rinda.
2. **Fine-tune sobre datos del dominio (SUP-12).** Vía correcta para resolver la
   clasificación CAEX/dozer y mitigar el recall nocturno. El enunciado permite
   datasets públicos adicionales; no se implementó por restricción de tiempo.
3. **Reidentificación por apariencia.** Resuelve la fragmentación de IDs y
   limpia el conteo de alertas.
4. **Unidad normalizada de reporte (SUP-16).** Desacopla la salida del supuesto
   de altura de SUP-07. De bajo costo y alto valor.

---

## Entorno de desarrollo

Desarrollado en Apple Silicon (MPS). La imagen se construye con
`--platform linux/amd64` para el despliegue objetivo x86_64 con CUDA. El
dispositivo se resuelve en cascada `cuda → mps → cpu`: el fallback a CPU es
obligatorio (SUP-23), ya que un contenedor que entrega artefactos lentamente
vale más que uno que aborta por ausencia de GPU.

Las cifras de rendimiento publicadas (5–8 FPS) corresponden a ejecución en CPU
sobre runner de CI. En el hardware objetivo serán sustancialmente mejores.
