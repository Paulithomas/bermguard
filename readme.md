# BermGuard AI

Pipeline de visión artificial para inspección de pretiles de seguridad y
monitoreo de proximidad de maquinaria en botaderos mineros.

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
las rutas por defecto.

`--method` acepta `1`, `2` o `all`. Con `all` se generan subdirectorios
separados por método más un resumen comparativo.

## Artefactos generados

Por cada video de entrada se crea un subdirectorio con:

| Archivo | Contenido |
|---|---|
| `<video>_osd.mp4` | Video con detecciones, semáforo de proximidad, delineación de pretil y panel de métricas |
| `berm_height.png` | Curva temporal de altura del pretil, cruda y suavizada, con tramos sin dato marcados |
| `vehicle_distribution.png` | Mapa de posiciones de maquinaria por nivel de riesgo y permanencia por equipo |
| `metadata.json` | Métricas de inferencia, estadísticos de altura y trazabilidad de la calibración |

Los cuatro artefactos se escriben siempre, incluso ante detección nula. Un
resultado nulo bien documentado es un resultado; un directorio vacío es un
fallo de ingeniería.

## Arquitectura
main.py CLI y validación de argumentos
src/
├── detection/ base.py define el contrato que implementan los métodos
├── berm/ extracción de cresta y estimación de altura
├── geometry/ calibración métrica y cálculo de proximidad
├── osd/ renderizado del On-Screen Display
├── analytics/ generación de gráficos
├── utils/ resolución de dispositivo de cómputo
└── pipeline.py orquestación
configs/default.yaml umbrales y parámetros, sin recompilar
docs/SUPUESTOS.md


`VehicleDetector` es una clase abstracta: agregar un método nuevo significa
implementar esa interfaz y registrarlo en la fábrica, sin tocar el orquestador.

## Decisiones de ingeniería

Las decisiones adoptadas ante las ambigüedades del enunciado están registradas
en **[docs/SUPUESTOS.md](docs/SUPUESTOS.md)**, con identificadores estables
`SUP-NN` referenciados desde el código, la configuración y este documento.

Las más relevantes:

**Escala métrica sin calibración de cámara (SUP-15).** No hay parámetros
intrínsecos disponibles. La escala se resuelve por metrología de vista única
usando la maquinaria como referencia de altura conocida, mediante una cascada
de tres niveles que declara siempre su origen en `metadata.json`. Validado
contra marcado manual: error de 1.4% en la estimación del horizonte y 5.4% en
la altura de cámara.

**El material es sintético y carece de ground truth (SUP-11).** No existe
verdad física contra la cual validar. El objetivo declarado es trazabilidad y
consistencia, no exactitud absoluta. La dispersión de la estimación de escala
se mide y se reporta.

**La cámara no es fija (SUP-10).** El supuesto inicial fue refutado
empíricamente: se midió deriva continua de 1-2 px por frame. Se descartaron en
consecuencia el fondo por mediana temporal, el ROI estático y la sustracción de
fondo.

## Limitaciones conocidas

**Clasificación CAEX/dozer (SUP-29).** COCO no contiene clase para maquinaria
de oruga; ambos vehículos se detectan como `truck`. Se evaluaron y descartaron
dos discriminantes sin datos etiquetados: la relación de aspecto (un CAEX de
perfil alcanza ratio 1.98, indistinguible del 1.94 de un dozer) y la
consistencia física de la altura implicada (ambigüedad geométrica genuina de la
vista monocular). Se conserva la clase única en la línea base.

**Selección de la cresta del pretil (SUP-01).** El extractor detecta
estructuras horizontales de forma estable, pero el perfil de energía presenta
varios picos bien formados —talud de fondo, cresta, ripio de primer plano— y el
criterio de máxima energía no siempre selecciona el pretil correcto.

**Recall nocturno (SUP-30).** El método 1 cae de 0.88 a 0.39 detecciones por
frame en escenas nocturnas con faros directos. Se verificó visualmente la
presencia de al menos tres vehículos en cuadro frente a 0.39 detectados. El
preprocesado CLAHE recupera un 45%, insuficiente: la causa es ausencia de
información local por saturación, no falta de contraste global.

**Segmentación de tomas (SUP-09).** Se detectan 12 de 16 cortes reales. Los no
detectados son transiciones graduales de 3 a 6 frames, no cortes duros.

## Trabajo pendiente

- **Fine-tune sobre datos del dominio.** Es la vía correcta para resolver la
  clasificación CAEX/dozer y el recall nocturno. El enunciado permite datasets
  públicos adicionales; no se implementó por restricción de tiempo.
- **Desambiguación temporal de la cresta.** El pretil es estático entre frames
  mientras que huellas y sombras se desplazan. Acumular la respuesta de textura
  sobre varios frames compensados por movimiento debería hacer emerger la
  estructura real.
- **Persistencia de IDs de tracking.** La detección intermitente genera IDs
  nuevos al reaparecer un vehículo, lo que infla el conteo de equipos.

## Entorno de desarrollo

Desarrollado en Apple Silicon (MPS). La imagen se construye con
`--platform linux/amd64` para el despliegue objetivo x86_64. El dispositivo se
resuelve en cascada `cuda → mps → cpu`: el fallback a CPU es obligatorio
(SUP-23), ya que un contenedor que entrega artefactos lentamente vale más que
uno que aborta por ausencia de GPU.