# Registro de Supuestos de Ingeniería — BermGuard AI

**Documento:** `docs/SUPUESTOS.md`
**Versión:** 1.1
**Estado:** vigente — supuestos duros validados o refutados empíricamente (ver §7)

---

## 1. Propósito

El enunciado del desafío establece:

> *"¿Dudas técnicas o ambigüedades? Asume el criterio de ingeniería más sólido, seguro y justificado, documentando tus decisiones en el README.md."*

Este documento es la respuesta formal a ese mandato. Registra **todos** los supuestos adoptados para resolver las ambigüedades del enunciado y las limitaciones del material entregado.

El principio rector es que **en ausencia de ground truth, un supuesto declarado y auditable vale más que un número presentado como verdad**. Cada medición que produce este sistema es trazable hasta el supuesto que la sostiene.

### Cómo leer este documento

- **Tipo Duro:** si el supuesto resulta falso, el diseño requiere modificación estructural.
- **Tipo Blando:** si resulta falso, degrada la calidad del resultado pero el sistema sigue operando y reportando.
- **Estado Validado:** verificado empíricamente sobre el material de muestra, con el artefacto que lo respalda.
- **Estado Refutado:** la medición contradijo la formulación original. Se conserva el registro del supuesto refutado junto a su reformulación, porque el proceso de descarte es parte del resultado.
- **Estado Diseñado, no implementado:** el criterio está definido y en algunos casos medido, pero no está integrado al pipeline de inferencia. Se declara explícitamente en lugar de presentarse como vigente.

### Distinción entre medido, adoptado e implementado

Este documento separa deliberadamente tres cosas que suelen confundirse:

- **Medido:** existe un experimento con resultado numérico y artefacto asociado.
- **Adoptado:** se tomó una decisión de diseño a partir de esa medición.
- **Implementado:** la decisión está materializada en el código que corre en producción.

Un supuesto puede estar medido y adoptado sin estar implementado. Los casos en que esto ocurre están señalados en cada supuesto y consolidados en §7.

### Convención de referencia cruzada

Cada supuesto tiene un ID estable `SUP-NN`. Los IDs se referencian desde:

- `configs/default.yaml` — como comentario sobre el parámetro que materializa el supuesto.
- Docstrings de los módulos que lo implementan.
- `docs/reporte_benchmark.md` — al discutir limitaciones de las métricas.
- `docs/EXPERIMENTOS.md` — bitácora de enfoques evaluados y descartados.
- `docs/CONSISTENCIA_TEMPORAL.md` — benchmark de la alternativa temporal a SUP-01.
- `metadata.json` — vía el campo `assumptions_version`.

---

## 2. Supuestos de dominio operativo

### SUP-01 — Definición operativa de "pretil"
**Tipo:** Duro · **Estado:** Implementado, con limitación medida · **Módulo:** `src/berm/crest_extractor.py`

Se define el pretil como **la cresta continua de mayor extensión horizontal dentro del plano operativo de descarga**, sujeta a tres filtros de exclusión:

1. Debe estar por debajo de la línea de horizonte estimada (excluye cordillera y cerros de fondo).
2. Su altura proyectada debe ser inferior a 8 m (excluye taludes de botadero lejanos y bancos del rajo, que no son estructuras de contención).
3. Debe presentar continuidad horizontal mínima del 20% del ancho de imagen.

Si el bulldozer ocluye un tramo de la cresta, sus píxeles se excluyen mediante la máscara del detector.

Si ninguna candidata supera los filtros, el sistema reporta `berm_detected: false` para esa toma. **Esto es un resultado válido, no un fallo.**

**Justificación.** El material de muestra presenta tres o más candidatos a "pretil" en una misma escena: camellones bajos en primer plano, montículos intermedios y la berma de descarga al fondo. El enunciado no desambigua. Sin un criterio explícito la métrica no es reproducible ni auditable.

**Limitación medida.** La selección de la fila de cresta usa el criterio de máxima energía por frame, que es una decisión independiente entre frames. El modo de fallo no es ruido sino **mesetas discretas y sostenidas** en la serie de altura: cuando el perfil de energía tiene varios picos bien formados, el máximo se ancla a una estructura distinta y permanece ahí. Afecta a video_01 y video_02; video_03 y video_04 presentan curvas continuas.

**Alternativa evaluada, no integrada.** Se implementó y midió un trazador de Viterbi con prior de transición sobre coordenadas compensadas por deriva. Reduce la desviación absoluta mediana entre 25% y 82% según el video, y la tasa de saltos espurios entre 75% y 95%, a cambio de 6% a 25% de energía. **No está integrada al pipeline.** Detalle metodológico y hallazgos en `docs/CONSISTENCIA_TEMPORAL.md`.

---

### SUP-02 — Modelo de rasante
**Tipo:** Duro · **Estado:** Implementado · **Módulo:** `src/geometry/calibration.py`

La rasante es el **plano del suelo transitable del lado de la cámara**, aproximado como plano localmente en la zona de interés.

**Justificación.** Es la referencia que el enunciado exige explícitamente ("su base respecto al nivel de la rasante/suelo transitable"). La hipótesis de planaridad local es la única que habilita metrología monocular sin información de profundidad.

---

### SUP-03 — Formulación de la altura
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `src/berm/crest_extractor.py`, `src/pipeline.py`

La altura se calcula **por columna**:

```
H(x, t) = (y_base(x, t) − y_cresta(x, t)) / px_por_m(y)
```

y se reporta por frame como **mediana con percentiles p10 y p90**, no como escalar único.

**Justificación.** El pretil no tiene altura uniforme a lo largo de su extensión. Un escalar único oculta precisamente la información que un fiscalizador necesita: dónde el pretil está bajo norma. La dispersión es parte del resultado.

**Nota de lectura.** En video_01 y video_02 el p90 más que duplica la mediana (6.06 contra 2.75 y 6.17 contra 2.69). Esa asimetría no describe un pretil irregular: es la cola que produce el modo de fallo de SUP-01. En video_03 y video_04, sin ese fallo, el p90 queda contenido (2.61 y 2.57).

---

### SUP-04 — Umbrales de proximidad
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `configs/default.yaml`

| Nivel | Rango | Color OSD |
|---|---|---|
| Seguro | > 20 m | `#10B981` (verde) |
| Precaución | 10 – 20 m | `#F59E0B` (amarillo) |
| Alerta crítica | < 10 m | `#EF4444` (rojo) |

Parametrizables sin recompilar la imagen.

**Justificación.** El PDF los menciona como ejemplo cualitativo; el brief HTML los fija numéricamente junto al hex exacto y al campo `proximity_alert`. Se adopta el brief por ser la fuente más específica.

---

### SUP-05 — Métrica de distancia entre equipos
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `src/geometry/distance.py`

Distancia euclidiana en el plano de rasante entre los **puntos de contacto con el suelo** de cada equipo (centro inferior del bounding box), proyectados mediante la calibración de la toma. Se registra la distancia mínima por par.

**Limitación declarada.** La medición centro a centro **sobreestima** la separación real entre carrocerías, dado que un CAEX mide del orden de 15 m de largo. Se reporta como **cota superior conservadora**; la distancia real entre superficies es menor. En un despliegue productivo correspondería usar la separación entre envolventes convexas proyectadas.

---

### SUP-06 — Taxonomía de clases
**Tipo:** Blando · **Estado:** Implementado con limitación (ver SUP-29) · **Módulo:** `src/detection/`

| Clase | Descripción |
|---|---|
| `caex` | Camión de extracción minera |
| `dozer` | Bulldozer / tractor de oruga |
| `other_heavy` | Otra maquinaria pesada |

`other_heavy` **participa del cálculo de proximidad** pero no se contabiliza en las métricas de clasificación del benchmark.

**Justificación.** El enunciado nombra explícitamente CAEX y bulldozers, pero también se refiere a "maquinaria de apoyo" en general. Ignorar un equipo no clasificado en el cálculo de proximidad sería un error de seguridad.

---

### SUP-07 — Convención métrica de referencia
**Tipo:** Duro · **Estado:** Validado (recon 07) e implementado · **Módulo:** `configs/default.yaml`

| Equipo | Altura nominal | Ancho nominal |
|---|---|---|
| CAEX clase 300 t | 7.4 m | 9.0 m |
| Bulldozer clase D11 | 4.5 m | — |

**Toda medida expresada en metros por este sistema es relativa a esta convención declarada.**

**Justificación.** Es el ancla de la que depende toda la analítica métrica. Al declararla como *convención* en lugar de presentarla como *verdad medida*, la salida del sistema pasa de ser una estimación con error desconocido a ser una medición auditable bajo supuesto explícito. Un tercero puede recalcular cualquier resultado con otra convención sin reprocesar el video.

**Validación.** Con CAEX = 7.4 m, la altura de cámara estimada en video_01 resulta 11.95 m ± 0.30 m, valor físicamente plausible para un poste de faena. La convención produce geometría coherente; no se valida su exactitud, que no es verificable sobre material sintético (SUP-11).

**Advertencia de propagación.** `src/geometry/calibration.py` usa exclusivamente `caex_height_m` como referencia; `dozer_height_m` solo interviene en el clasificador. Un dozer tomado por CAEX introduce un sesgo sistemático de factor 7.4/4.5 en la escala de esa toma. Ver SUP-29.

---

## 3. Supuestos sobre el material de entrada

### SUP-08 — Duración y presupuesto computacional
**Tipo:** Blando · **Estado:** Validado (recon 01)

Los videos duran aproximadamente 10 s. El costo computacional **no es restrictivo** en modo batch.

**Medición.** 4 videos, 1.022 frames totales. video_01: 1920×1080 @ 30 fps, 302 frames. video_02/03/04: 1280×720 @ 24 fps, 240 frames cada uno. Códec H.264 en todos.

**Consecuencia de diseño.** Habilita el uso de modelos de alta latencia sin penalización práctica. Se sigue midiendo y reportando FPS, pero se interpreta en el contexto correcto: 5 FPS es aceptable en procesamiento batch offline e inaceptable en un sistema de seguridad en tiempo real. Esa distinción se discute en el reporte de benchmark.

---

### SUP-09 — Un archivo puede contener múltiples tomas
**Tipo:** Duro · **Estado:** Validado empíricamente · **Implementado solo en reconocimiento** · **Módulo:** `tools/recon/02_shots.py`, `tools/recon/04_orb_shots.py`

Un archivo de video **no equivale a una toma continua**. Puede contener varios cortes de escena, incluyendo cambios de iluminación dentro del mismo archivo.

**Ground truth.** 16 cortes reales verificados manualmente reproduciendo los cuatro videos: 5 en video_01, 3 en video_02, 4 en video_03, 4 en video_04. Ocurren en múltiplos aproximados de 2 s, consistente con concatenación de clips generativos independientes.

**Detectores evaluados.**

| Detector | Parámetro | Cortes detectados |
|---|---|---|
| `ContentDetector` | threshold 27 / 40 / 55 | 6 (1 falso positivo) / 5 / 4 |
| `AdaptiveDetector` | 3.0 | 6 |
| `AdaptiveDetector` | 1.5 / 1.0 / 0.7 | 12 (saturado) |
| ORB, tasa de coincidencia | drop 0.45 | 5 |

**Diagnóstico.** El detector adaptativo satura en 12: bajar el parámetro de 1.5 a 0.7 no cambia el resultado. La serie temporal de coincidencia ORB explica por qué: en un corte duro la tasa cae a 0.002, mientras que en los cortes no detectados solo baja a ~0.47 sobre una mediana de 0.71. Los 4 no detectados son **transiciones graduales de 3 a 6 frames**, y ningún método basado en discontinuidad entre frames consecutivos puede detectarlos.

**Decisión adoptada.** Unión de ambos detectores. Se acepta que una toma detectada pueda contener más de una escena real, siempre de condición lumínica similar.

> **Estado de implementación.** La detección de tomas opera únicamente en las herramientas de reconocimiento, y sus resultados están versionados en `data/recon/shots.json` y `data/recon/orb_cuts.json`. **El pipeline de inferencia procesa cada archivo como una unidad única**: calibración, tracking y filtrado temporal no se segmentan por toma. La segmentación dentro del pipeline es trabajo pendiente y es el prerrequisito de SUP-19.

**Evidencia adicional (benchmark temporal).** El filtro de plausibilidad de deriva de `tools/recon/08_crest_viterbi.py` descarta pasos donde la homografía devuelve más de 10 px de desplazamiento, algo imposible dentro de una escena continua con deriva de 1–2 px por frame. El conteo de descartes se comporta como detector independiente de discontinuidades:

| Video | Descartes | Cortes declarados en artefactos |
|---|---|---|
| 01 | 3 / 301 | 5 |
| 04 | 12 / 239 | 0 |
| 02 | 18 / 239 | 1 |
| 03 | 21 / 239 | 0 |

Sugiere que hay más discontinuidades de las que los detectores actuales declaran, y ofrece una vía concreta de mejora. Detalle en `docs/CONSISTENCIA_TEMPORAL.md` §5.3.

**Consecuencia crítica.** SUP-09 es prerrequisito de SUP-01: el prior temporal de selección de cresta solo rinde cuando opera dentro de una escena continua. Video_01, con cortes bien detectados, alcanza 82% de reducción de MAD; video_03, sin cortes declarados, se queda en 25%.

---

### SUP-10 — Deriva de cámara dentro de la toma
**Tipo:** Duro · **Estado:** REFUTADO y reformulado (recon 05)

**Formulación original.** Dentro de cada toma la cámara es fija, con deriva residual inferior al 2% del ancho de imagen entre el primer y último frame.

**Resultado empírico: falso.** Solo 1 de 6 tomas cumple el criterio. La cámara presenta deriva continua tipo dolly/paneo.

**Caracterización.** Homografía ORB+RANSAC sobre ventanas de 10 frames: escala entre 0.984 y 0.996 (variación bajo 1.6%), desplazamiento del centro entre 4 y 19 px. Equivale a 1–2 px por frame. Las ventanas con escala 1.097 y desplazamiento de 117 px corresponden a transiciones de escena no segmentadas (SUP-09), no a deriva.

**Reformulación adoptada.** La cámara presenta deriva lenta continua. No se asume marco de referencia estático en ningún punto del pipeline.

**Consecuencias de diseño.** Se descartan cuatro técnicas que dependían del supuesto original: fondo por mediana temporal, ROI estático, sustracción de fondo MOG2 y calibración única por toma. Ninguna de ellas está en el código, que es precisamente la consecuencia de haber medido antes de implementar.

**Nota metodológica.** La primera medición de deriva arrojó valores de hasta 28% con solo 8–10 inliers por homografía. Se descartó por baja confianza: con tan pocas correspondencias, RANSAC encuentra alguna transformación que encaja sin que ello signifique nada. La medición válida se obtuvo sobre ventanas cortas, con 30 inliers mínimos y cobertura declarada. Este criterio se reutiliza en el benchmark temporal.

---

### SUP-11 — Ausencia de ground truth físico
**Tipo:** Duro · **Estado:** Declarado en el enunciado

El material es **sintético generado por IA** y **no satisface consistencia proyectiva estricta**. No existe cámara real, no existen parámetros intrínsecos, y la escena no es rígida entre frames.

**No existe ground truth físico contra el cual validar la altura del pretil.** No se trata de que no esté disponible: no existe.

**Consecuencias que atraviesan todo el diseño:**

1. La dispersión de escala entre frames es **inherente al material**, no error del algoritmo. Se mide y se reporta (SUP-17), no se oculta.
2. Se esperan geometrías físicamente imposibles en la maquinaria. El detector se configura en consecuencia (SUP-18).
3. El objetivo del sistema deja de ser exactitud métrica y pasa a ser **trazabilidad y consistencia declarada**.

**Nota interpretativa.** El enunciado señala que *"no se logre el fin completo de detectar todo el pretil o medirlo sin distorsión o vibración, no será excluyente"*. Se interpreta no como indulgencia, sino como reconocimiento de que parte de la vibración es irreducible por la naturaleza del material. El entregable de valor es el tratamiento riguroso de esa limitación.

**Matiz empírico (recon 07).** La consistencia proyectiva es mejor de lo previsto: 2.5% de dispersión en la estimación de escala intra-escena. La ausencia de ground truth sigue vigente, pero la inconsistencia del generador **no domina el error**. Esto traslada la responsabilidad del error a los supuestos del sistema, no al material.

---

### SUP-12 — Distribución del set ciego
**Tipo:** Blando · **Estado:** Pendiente

El set de testing oculto proviene del **mismo generador** y comparte distribución con los videos de muestra.

**Consecuencia de diseño.** Justifica hacer fine-tune sobre frames sintéticos, práctica que en otro contexto constituiría un error metodológico. Se mitiga el riesgo de sobreajuste a artefactos del generador mezclando datos reales de fuentes públicas en el conjunto de entrenamiento. **No se realizó fine-tune** por restricción de tiempo.

---

## 4. Supuestos geométricos y de calibración

### SUP-13 — Modelo de cámara
**Tipo:** Blando · **Estado:** Adoptado por necesidad

Modelo pinhole sin corrección de distorsión radial.

**Justificación.** No hay intrínsecos disponibles ni posibilidad de calibración con patrón. Adicionalmente, la distorsión aparente en material generativo no sigue un modelo físico consistente, por lo que aplicar una corrección paramétrica introduciría error en lugar de removerlo.

---

### SUP-14 — Estimación de la línea de horizonte
**Tipo:** Duro · **Estado:** Validado con método alternativo (recon 07) e implementado · **Módulo:** `src/geometry/calibration.py`

**Formulación original.** El horizonte se estima por punto de fuga de líneas paralelas del plano de suelo — huellas de neumático, bordes de camino, línea de base del pretil.

**Resultado: inviable.** Los caminos del material son curvos. Los dos bordes de un camino en curva no son paralelos en el mundo real, por lo que su intersección en la imagen no corresponde al punto de fuga. El resultado cayó fuera de la imagen (y = 1434 sobre un frame de 1080 px), lo que hizo evidente el error.

**Método adoptado.** Intersección de la línea de bases con la línea de cimas de dos vehículos de altura conocida situados a distinta profundidad. Horizonte estimado en y ≈ 489 ± 4 px sobre 1080 en video_01, estable entre frames.

**Se mantiene la exclusión de la silueta de montañas.** La cordillera de fondo está *por encima* del horizonte verdadero. Usarla como referencia introduce un sesgo sistemático que se propaga a toda la metrología.

**Fallback.** Si no se logra estimar el horizonte, se asume `h/2` y se marca la calibración con confianza baja.

---

### SUP-15 — Cascada de resolución de escala
**Tipo:** Duro · **Estado:** Validado e implementado (niveles 1 y 2 observados en producción) · **Módulo:** `src/geometry/calibration.py`

La escala métrica se resuelve mediante metrología de vista única sobre el plano de suelo:

```
H_objeto = h_cámara × (y_base − y_top) / (y_base − y_horizonte)
```

con `y` creciendo hacia abajo. Verificación de coherencia: si la cima de un objeto coincide con el horizonte, `H = h_cámara`.

La resolución sigue una cascada de tres niveles, **declarada siempre en `metadata.json`**:

| Nivel | `scale_source` | Condición | Confianza |
|---|---|---|---|
| 1 | `vehicle_reference` | Hay vehículo apoyado y confiable en la toma. Se invierte la fórmula con la altura nominal de SUP-07 para despejar `h_cámara` | Alta |
| 2 | `assumed_camera_height` | No hay vehículo. Se asume altura de poste de faena en rango 8–12 m | Media |
| 3 | `none` | No se puede estimar horizonte ni referencia. Se reporta solo en unidades normalizadas | Nula |

**Justificación.** El material de muestra incluye tomas sin ningún vehículo en cuadro. Un sistema que dependa exclusivamente del nivel 1 falla en el set ciego. La cascada garantiza salida útil y honestamente etiquetada en todos los casos.

**Validación contra marcado manual (recon 07).** Sobre video_01: horizonte automático 495.8 contra 489 manual (error 1.4%), y h_cámara 11.30 m contra 11.95 m (error 5.4%), con dispersión interna 3.6% y confianza 0.93. Sobre video_03: horizonte 388.2 contra 395.8 (error 1.9%), dispersión 1.1%, confianza 0.98. El nivel 1 opera sin intervención humana.

**Validación de la referencia por bbox de detección.** Sobre video_01 f30, dos CAEX a distancias muy distintas (232 px y 47 px de altura aparente) arrojan h_cámara = 11.62 m y 11.63 m, con desviación de 1 cm. Coincide con los 11.95 m del marcado manual dentro de 2.7%. La referencia automática es viable.

**Corolario: la altura de cámara no es constante entre videos.** video_01 arroja ≈11.6 m con horizonte en y = 489; video_03 arroja ≈5.4 m con horizonte en y = 395.8. Son tomas independientes, con resoluciones y framerates distintos (SUP-28). **La calibración debe ser por toma, sin valor de referencia compartido.**

**Advertencia metodológica.** En video_03 la altura estimada queda bajo el rango típico de un poste de faena, porque el bulldozer entra al cálculo asumido como CAEX (SUP-29). La dispersión baja no lo detecta: el sesgo es sistemático, no aleatorio. **Una dispersión baja indica consistencia interna, no exactitud.**

**Degradación observada en producción.** En video_04 la cascada no encontró referencia vehicular suficiente y degradó al nivel 2 (`assumed_camera_height`, confianza 0.40), declarándolo explícitamente en lugar de emitir un número sin respaldo. Es el comportamiento previsto por el diseño, verificado sobre material real.

---

### SUP-16 — Reporte en doble unidad
**Tipo:** Blando · **Estado:** Diseñado, no implementado

Toda medida de altura y proximidad debería reportarse en dos unidades:

- `height_m` — metros bajo la convención SUP-07.
- `height_normalized` — en unidades de altura-CAEX (adimensional).

**Justificación.** La unidad normalizada es **inmune al error de escala** y sigue satisfaciendo el objetivo operativo real, que según el enunciado es verificar que *"la altura de este pretil debe mantenerse constante"*. La constancia es detectable sin conocer la escala absoluta.

> **Estado de implementación.** `metadata.json` reporta únicamente metros. La unidad normalizada no está implementada. Es trabajo pendiente de bajo costo y alto valor, dado que desacopla el reporte del supuesto de SUP-07.

---

### SUP-17 — Cuantificación de la incertidumbre de escala
**Tipo:** Blando · **Estado:** Validado (recon 07) · **Implementado parcialmente** · **Módulo:** `src/geometry/calibration.py`

`h_cámara` se estima en los frames muestreados de la toma que contengan un vehículo apoyado confiable. Su **dispersión** se adopta como medida de la incertidumbre inherente y se declara en `metadata.json` como `dispersion_pct`.

**Medición (recon 07).** Dispersión de h_cámara = 2.5% (media 11.95 m, desviación 0.30 m, n = 3) sobre frames 10/20/30 de la toma diurna de video_01.

**Alcance de esa validación.** Los tres frames pertenecen a la misma toma y abarcan 20 frames. El resultado mide **consistencia intra-escena**.

**Dispersión medida en producción (método 1).** video_02: 5.9%. video_01: 6.9%. video_03: 11.8%. video_04: no aplica (nivel 2, sin dispersión calculable).

**Regla de decisión sobre la unidad primaria:**

| Dispersión de `h_cámara` | Decisión |
|---|---|
| < 10 % | Metros como unidad primaria, confianza moderada |
| 10 – 25 % | Metros con banda de incertidumbre obligatoria en todo reporte |
| > 25 % | Unidad normalizada pasa a primaria; metros queda como estimación secundaria rotulada |

> **Estado de implementación.** La dispersión se calcula y se reporta, pero la **banda de incertidumbre no se dibuja** en los gráficos de altura, y la regla de decisión sobre la unidad primaria no se aplica automáticamente. Video_03, con 11.8%, cae en la banda que exigiría banda de error obligatoria. Depende de SUP-16.

---

## 5. Supuestos de modelado y estabilidad temporal

### SUP-18 — Detector permisivo con filtrado fuerte
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `src/detection/`, `configs/default.yaml`

Se prefiere un detector con **umbral de confianza bajo** acompañado de filtrado temporal fuerte, por sobre un detector estricto sin filtrado.

**Justificación.** Derivado de SUP-11: las geometrías imposibles del material generativo penalizan desproporcionadamente a modelos rígidos entrenados sobre maquinaria real. El tracker asume la responsabilidad de rechazar falsos positivos mediante consistencia temporal, criterio más robusto en este material que la confianza puntual del detector.

---

### SUP-19 — Reinicialización de estado en cortes de toma
**Tipo:** Duro · **Estado:** Diseñado, no implementado

Los IDs de tracking y los filtros temporales **deberían reinicializarse en cada corte de toma**.

**Justificación.** Derivado directo de SUP-09. Filtrar a través de un corte de escena produce artefactos considerablemente peores que no filtrar en absoluto: la señal resultante mezcla geometrías incompatibles y no representa a ninguna de las dos tomas.

> **Estado de implementación.** `src/pipeline.py` llama `detector.reset_tracker()` **una sola vez por video**, antes de iniciar el bucle de frames. No hay reinicialización en los cortes. En consecuencia, el tracking y la mediana móvil de altura atraviesan las transiciones de escena. Es consecuencia directa de que SUP-09 no esté implementado en el pipeline, y comparte con él la misma tarea pendiente.

**Efecto observable.** Contribuye a la inflación del conteo de equipos por reaparición de IDs, registrada en el trabajo pendiente del README.

---

### SUP-20 — El suavizado es un prior, no un denoiser
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `src/analytics/plots.py`

El suavizado temporal sobre `H(t)` opera como **prior de rigidez**: le impone a la escena una consistencia física que la escena no posee. No es eliminación de ruido de medición sobre una señal real subyacente.

**Consecuencia obligatoria.** Se exportan **ambas curvas** — cruda y filtrada — superpuestas en el mismo gráfico, junto con los tramos sin dato marcados explícitamente.

**Justificación.** Derivado de SUP-11. Presentar únicamente la curva filtrada equivale a presentar el supuesto como si fuera observación. La curva cruda es la evidencia; la filtrada es la interpretación.

---

### SUP-21 — Concordancia entre métodos como proxy de confiabilidad
**Tipo:** Blando · **Estado:** Validado · **Módulo:** `docs/reporte_benchmark.md`

Ante la ausencia de ground truth (SUP-11), el **grado de acuerdo entre los dos métodos implementados** se adopta como proxy de confiabilidad de la estimación.

**Justificación.** Se dispone de dos estimadores independientes. Su convergencia o divergencia es información legítima y no requiere etiquetas. Esto transforma el requisito del Módulo 1 de una comparación cosmética en un **mecanismo de validación cruzada** que sustituye parcialmente al ground truth ausente.

**Medición.** Tres de los cuatro videos convergen en altura mediana de pretil bajo el 6% de diferencia entre métodos. La divergencia del cuarto es explicable: es el único donde ambos métodos usaron niveles de calibración distintos.

> **Nota.** La concordancia se calcula en el reporte de benchmark a partir de los `metadata.json` de ambos métodos. No existe un `comparison.json` generado automáticamente por `--method all`.

---

### SUP-22 — Preprocesado adaptativo por toma
**Tipo:** Blando · **Estado:** Medido y adoptado; **no implementado**

La clasificación de iluminación (día / crepúsculo / noche) debería realizarse **a nivel de toma**, a partir del brillo medio y del histograma, activando el preprocesado correspondiente por toma.

**Justificación.** Activar el preprocesado frame a frame genera parpadeo visible en el OSD. Aplicarlo siempre degrada innecesariamente las tomas diurnas bien expuestas.

> **Estado de implementación.** No existe módulo de preprocesado en el pipeline. La clasificación de iluminación por toma está registrada en `data/recon/shots.json` (campo `lighting`) como producto del reconocimiento, pero no se consume en inferencia. El CLAHE evaluado en SUP-30 **no está aplicado**. Depende de SUP-09.

---

## 6. Supuestos de despliegue y ejecución

### SUP-23 — Disponibilidad de GPU y degradación a CPU
**Tipo:** Duro · **Estado:** Validado e implementado · **Módulo:** `src/utils/device.py`

Se asume ejecución con `--gpus all` sobre GPU NVIDIA/CUDA. El sistema **detecta la ausencia de CUDA y degrada automáticamente a CPU sin fallar**, mediante cascada `cuda → mps → cpu`.

**Justificación.** El comando de ejemplo del PDF no incluye `--gpus`; el del brief HTML sí. El fallback cubre ambos escenarios. Un contenedor que entrega artefactos lentamente en CPU vale sustancialmente más en la evaluación que uno que aborta por ausencia de dispositivo.

**Validación.** Ejecución completa sobre runner Linux x86_64 sin GPU (GitHub Actions): los cuatro videos procesados, los cuatro artefactos escritos por video, a 4.2–5.5 FPS. La degradación opera sin intervención.

---

### SUP-24 — Ausencia de red en runtime
**Tipo:** Duro · **Estado:** Implementado; validado en build, pendiente la prueba `--network none` · **Módulo:** `Dockerfile`, `tools/prepare_weights.py`

**No hay conectividad de red durante la ejecución.** Todos los pesos se descargan durante `docker build` y quedan horneados en la imagen bajo `/app/weights`. Las variables `TORCH_HOME`, `YOLO_CONFIG_DIR` y `HF_HOME` se fijan dentro del contenedor para impedir que cualquier librería intente resolución remota.

**Justificación.** Requisito "Plug & Play" explícito del enunciado.

**Hallazgo de despliegue.** YOLO-World descarga el encoder de texto CLIP la primera vez que se invoca `set_classes()`, es decir **después** de cargar los pesos del detector. Copiar únicamente el `.pt` no basta: el contenedor fallaría al arrancar en un entorno sin red. Se resuelve forzando la inicialización del vocabulario durante el build. El encoder pesa 354 MB, catorce veces más que el detector que lo usa.

**Criterio de aceptación pendiente.** El contenedor debe completar un procesamiento íntegro bajo `docker run --network none`. Esta prueba forma parte del checklist de entrega.

---

### SUP-25 — Tolerancia a entradas malformadas
**Tipo:** Duro · **Estado:** Implementado · **Módulo:** `src/pipeline.py`

Archivos ilegibles, corruptos o de formato no soportado se **registran en log y se omiten sin abortar el procesamiento del lote**. `process_video` lanza `ValueError` ante un archivo ilegible y `run_pipeline` lo captura, contabiliza el fallo y continúa con el siguiente archivo.

**Justificación.** La evaluación es automatizada sobre un set ciego. Una excepción no capturada en el primer archivo pierde todos los videos restantes y con ello la totalidad de la evaluación.

---

### SUP-26 — Los artefactos se escriben siempre
**Tipo:** Duro · **Estado:** Validado e implementado · **Módulo:** `src/pipeline.py`, `src/analytics/plots.py`

Los cuatro artefactos requeridos se generan **para todo video procesado**, incluso ante detección nula. Los gráficos incluyen ruta de anotación explícita cuando no hay datos válidos.

**Justificación.** El enunciado acepta explícitamente resultado parcial o nulo. No acepta ausencia de entregable. Un resultado nulo bien documentado es un resultado; un directorio vacío es un fallo de ingeniería.

**Validación.** Los cuatro videos produjeron sus cuatro artefactos en la ejecución de CI, incluido video_04, cuya calibración degradó a nivel 2.

---

### SUP-27 — Semántica de `--method all`
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `main.py`, `src/pipeline.py`

`--method all` ejecuta ambos métodos y genera **subdirectorios separados por método**:

```
output/
└── video_01/
    ├── method_1/   (video OSD, gráficos, metadata.json)
    └── method_2/   (video OSD, gráficos, metadata.json)
```

**Justificación.** El enunciado pide "un argumento todo en 1". La separación por subdirectorio evita sobrescritura de artefactos homónimos y habilita el análisis de concordancia de SUP-21.

> **Nota.** El resumen comparativo `comparison.json` a nivel de video no está implementado. La comparación se realiza manualmente sobre ambos `metadata.json` en el reporte de benchmark.

**Corrección detectada por verificación de tipos.** `set_tracking` y
`reset_tracker` eran invocados por el pipeline sin estar declarados en
`VehicleDetector`. Un detector nuevo que implementara la interfaz completa
habría fallado en runtime. Se incorporaron a la clase base como métodos
concretos con implementación vacía, no abstractos: un detector sin estado
temporal satisface el contrato sin escribir código innecesario.

---

### SUP-28 — Heterogeneidad de resolución y framerate
**Tipo:** Duro · **Estado:** Validado (recon 01) e implementado

Los videos de entrada no comparten resolución ni framerate. El sistema no asume valores fijos.

**Consecuencias de diseño.** Todo parámetro geométrico se expresa relativo al ancho o alto de imagen, no en píxeles absolutos. El eje temporal de los gráficos se expresa en segundos, derivado del fps de cada archivo. La escala `px_por_m` se recalcula por video.

---

### SUP-29 — Ambigüedad estructural en la clasificación CAEX/dozer
**Tipo:** Duro · **Estado:** Validado empíricamente; sin solución adoptada · **Módulo:** `src/detection/vehicle_classifier.py`

COCO no contiene clase para maquinaria de oruga: CAEX y bulldozer se detectan ambos como `truck` (confianza 0.58–0.66 sobre el material de muestra).

**Discriminantes evaluados y descartados:**

1. **Relación de aspecto.** Insuficiente: un CAEX de perfil completo alcanza ratio 1.98, indistinguible del 1.94 de un dozer.
2. **Consistencia física de la altura implicada.** Insuficiente por **ambigüedad geométrica genuina** de la vista monocular: un vehículo más bajo y más cercano produce exactamente la misma señal que uno más alto y más lejano. Sobre video_03 f90, con horizonte medido en y = 395.8, ambos vehículos convergen a h_cámara ≈ 5.4 m bajo hipótesis CAEX, sin separación explotable. **No es un defecto de implementación.**
3. **Prompts de texto explícitos (método 2).** Sobre video_03 f90, con un CAEX y un bulldozer en cuadro y `"bulldozer"` / `"crawler tractor with blade"` en el vocabulario, ambos se etiquetaron `caex`. El vocabulario **sí produce la clase `dozer`**, pero de forma esporádica: en la corrida completa aparece con 1 a 2 frames de presencia en video_03 y video_04, frente a decenas de frames del mismo objeto etiquetado `caex`. **La clase existe en la salida pero no es estable sobre un mismo track**, de modo que no constituye una clasificación utilizable.

**Decisión.** Se conserva la clasificación única `caex` en la línea base. **Ningún enfoque zero-shot resuelve la clasificación en este material**; requiere fine-tune sobre datos del dominio (SUP-12), que no se realizó.

**Impacto acotado.** La lógica de proximidad no depende de la clase, y la calibración de escala usa el vehículo de mayor altura aparente, que es CAEX con alta probabilidad. El sesgo residual afecta a las tomas donde solo hay dozer en cuadro, y se manifiesta en la advertencia metodológica de SUP-15.

---

### SUP-30 — Degradación del método 1 en escenas nocturnas con faros
**Tipo:** Blando · **Estado:** Validado empíricamente; mitigación parcial vía método 2

El recall del método 1 cae drásticamente en escenas nocturnas con iluminación artificial directa. Sobre video_01, en detecciones por frame con conf = 0.25:

| Tramo | det/frame | Brillo medio |
|---|---|---|
| f0–46 (diurno) | 0.88 | 117 |
| **f46–116 (nocturno con faros)** | **0.39** | 67 |
| f116–141 | 3.62 | 131 |
| f141–171 | 2.70 | 133 |
| f171–215 | 1.87 | 134 |
| f215–302 | 2.48 | 85 |

**El brillo medio no explica el fallo.** El tramo f215–302 tiene brillo 85, comparable al tramo problemático, y rinde 2.48 det/frame. La causa es la naturaleza de la degradación, no su magnitud global.

**Diagnóstico visual (f80).** Se verifican al menos tres vehículos en cuadro contra 0.39 detectados. Los vehículos alejados aparecen como siluetas sin textura interna; los cercanos quedan envueltos en halos de saturación de los faros; el polvo en suspensión difumina los contornos.

**Preprocesado evaluado (23 frames del tramo f46–116):**

| Variante | det/frame |
|---|---|
| Sin preprocesado | 0.39 |
| CLAHE clip 2.0 | 0.48 |
| **CLAHE clip 4.0** | **0.57** |
| Gamma 1.6 | 0.35 |
| Gamma + CLAHE | 0.39 |

CLAHE clip 4.0 recupera un 45%, insuficiente. El resultado es consistente con el diagnóstico: **el problema no es contraste global sino ausencia de información local**, y ninguna transformación fotométrica recupera píxeles saturados.

> **Estado de implementación.** CLAHE clip 4.0 fue evaluado y **recomendado** sobre esta evidencia, pero **no está integrado al pipeline**. Su aplicación depende de SUP-22, que a su vez depende de SUP-09.

**Mitigación efectiva vía método 2.** La hipótesis registrada antes de implementar el método 2 —que un modelo fundacional degradaría menos al razonar sobre semántica visual amplia en lugar de patrones aprendidos de COCO— se confirmó: 1.35 det/frame contra 0.39, un incremento de 246%. Es la única mitigación realmente operativa. La limitación se reduce, no se resuelve.

---

### SUP-31 — Escala en el punto medio para distancias entre pares
**Tipo:** Blando · **Estado:** Implementado · **Módulo:** `src/analytics/plots.py`

La distancia métrica entre dos vehículos se calcula convirtiendo la distancia en píxeles entre sus puntos de contacto con el suelo (SUP-05), usando la escala `pixels_per_meter` evaluada en el **punto medio de sus coordenadas y**.

**Justificación.** La escala px/m depende de la profundidad en una vista en perspectiva, por lo que no existe un factor único válido para un par de vehículos situados a distinta distancia. El punto medio es la aproximación de primer orden correcta cuando ambos están sobre el mismo plano de suelo, y evita el sesgo de elegir arbitrariamente la escala de uno de los dos.

**Restricción de concurrencia.** Solo se comparan detecciones del mismo frame. Dos vehículos que ocuparon la misma posición en instantes distintos no estuvieron próximos. Los tracks sin ID estable se excluyen, ya que no pueden asociarse entre frames.

**Limitación de cobertura.** La matriz solo compara tracks con ID estable, de modo que un video con muchas alertas puede no producir ningún par trazable. En video_02 el método 2 registra 71 alertas de proximidad y la matriz queda vacía: las alertas se calculan sobre todas las detecciones del frame, mientras que la matriz exige identidad persistente. Ambas cifras son correctas y miden cosas distintas. La cobertura de la matriz depende por tanto de la calidad del tracking, no solo de la geometría.

**Uso como detector de fragmentación de tracks.** Un valor por debajo del ancho nominal del equipo (9 m para CAEX, SUP-07) es físicamente imposible entre dos vehículos distintos y delata que un mismo objeto recibió dos IDs. El método 1 lo produce en tres de los cuatro videos: 0.08 m en video_02 entre tracks de clases distintas, 1.49 m en video_01, y tres pares del mismo track bajo 3 m en video_04. La matriz, diseñada para reportar proximidad, resulta ser también una medición del problema de identidad descrito en el trabajo pendiente.

**Limitación.** La aproximación se degrada cuando la separación en profundidad entre ambos vehículos es grande, porque la escala varía de forma no lineal con `y`.

**Degradación.** Si la calibración no es métrica, la matriz de distancias mínimas se reporta en píxeles y lo declara en el título del gráfico.

---

## 7. Estado consolidado: medido, adoptado, implementado

La tabla separa las tres cosas para que ninguna afirmación de este documento pueda leerse como más fuerte de lo que es.

| ID | Medido | Adoptado | Implementado en el pipeline |
|---|---|---|---|
| SUP-01 | Sí (limitación y alternativa) | Criterio de máxima energía | Sí; la alternativa Viterbi **no** |
| SUP-07 | Sí (recon 07) | Sí | Sí |
| SUP-08 | Sí (recon 01) | Sí | n/a |
| **SUP-09** | **Sí (12/16 cortes)** | **Sí** | **No — solo en `tools/recon/`** |
| SUP-10 | Sí — **refutado** | Reformulado | Sí (no se asume cámara fija) |
| SUP-14 | Sí (recon 07) | Método alternativo | Sí |
| SUP-15 | Sí (recon 07 + producción) | Sí | Sí, niveles 1 y 2 observados |
| **SUP-16** | n/a | Sí | **No** |
| SUP-17 | Sí (recon 07 + producción) | Sí | Parcial: dispersión sí, banda de error no |
| **SUP-19** | n/a | Sí | **No — reset una vez por video** |
| SUP-20 | n/a | Sí | Sí |
| SUP-21 | Sí | Sí | Manual, sin `comparison.json` |
| **SUP-22** | Sí (clasificación lumínica) | Sí | **No** |
| SUP-23 | Sí (CI sin GPU) | Sí | Sí |
| SUP-24 | Build sí; `--network none` no | Sí | Sí |
| SUP-26 | Sí (4 videos) | Sí | Sí |
| SUP-27 | — | Sí | Sí, sin `comparison.json` |
| SUP-29 | Sí (4 enfoques descartados) | Clase única | Sí |
| **SUP-30** | Sí (CLAHE 4.0, +45%) | Sí | **No** |
| SUP-31 | n/a | Sí | Sí |

**Cadena de dependencia pendiente.** SUP-09 (segmentación por toma en el pipeline) es prerrequisito de SUP-19 (reinicialización de tracking) y de SUP-22 (preprocesado por toma), que a su vez es prerrequisito de aplicar el CLAHE de SUP-30. Es una sola tarea que desbloquea tres supuestos, y es la de mayor retorno del trabajo pendiente.

**Advertencia de reproducibilidad.** Las cifras de recon 07 citadas en SUP-07, SUP-14, SUP-15 y SUP-17 están documentadas aquí, pero los artefactos correspondientes en `data/recon/` fueron sobrescritos por ejecuciones posteriores: `scale_2veh.json` quedó sin frames y `scale.json` conserva únicamente un frame marcado `valid: false`, correspondiente al método de punto de fuga descartado en SUP-14. **Re-ejecutar esos scripts no reproduce las cifras citadas.** Regenerar los artefactos es tarea pendiente.

---

## 8. No-objetivos declarados

Se documentan explícitamente para que no sean interpretados como omisiones. Son **decisiones**, no descuidos.

**No se persigue exactitud métrica absoluta.** Por SUP-11 no existe verdad física contra la cual validar. El objetivo declarado es consistencia, trazabilidad y cuantificación de la incertidumbre.

**No se implementa dashboard ni interfaz web.** Excluido explícitamente por el enunciado.

**No se opera en tiempo real.** El diseño es batch offline. Se mide y reporta FPS, y el reporte de benchmark discute la brecha hacia un despliegue real de seguridad, donde la latencia sí sería un requisito duro.

**No se implementa cuantización INT8 ni TensorRT.** Con el volumen de frames involucrado (SUP-08) el beneficio práctico es nulo en modo batch. Se declara la técnica y el argumento de por qué no aporta en este escenario específico, en lugar de aplicarla sin discutir su pertinencia.

**No se resuelve la clasificación CAEX/dozer.** Cuatro enfoques evaluados y descartados (SUP-29). Requiere fine-tune con datos etiquetados del dominio, que no se realizó por restricción de tiempo.

**No se segmenta por toma dentro del pipeline.** La detección de tomas está construida y medida, pero su integración al pipeline de inferencia quedó fuera de alcance. Las consecuencias están declaradas en SUP-09, SUP-19, SUP-22 y §7.

---

## 9. Trazabilidad hacia los criterios de evaluación

| Criterio | Peso | Supuestos que lo sostienen |
|---|---|---|
| Robustez MLOps & Docker | 20 % | SUP-23, 24, 25, 26, 27 |
| Rigor Matemático & Analítica | 20 % | SUP-02, 03, 05, 07, 13, 14, 15, 16, 17, 31 |
| Performance Visual & Segmentación | 25 % | SUP-01, 09, 18, 19, 20, 22, 30 |
| Arquitectura & Benchmark | 35 % | Este registro en su totalidad, con énfasis en SUP-10, 11, 20, 21, 29, §7 y §8 |

El criterio de mayor ponderación evalúa, en palabras del enunciado, *"profundidad técnica y honestidad analítica en el reporte comparativo"*. Lo que materializa esa honestidad en este documento: SUP-10 registra un supuesto duro que la medición refutó; SUP-14 registra un método que falló y el alternativo que lo reemplazó; SUP-29 registra cuatro enfoques descartados; §7 distingue lo implementado de lo solamente adoptado y declara qué artefactos no reproducen sus propias cifras; y §8 explicita qué se decidió no hacer.

---

## 10. Bitácora de cambios

| Versión | Cambio |
|---|---|
| 1.0 | Versión inicial. SUP-01 a SUP-27 establecidos. Cinco supuestos duros pendientes de validación empírica. |
| 1.1 | Validación empírica de SUP-07, 08, 09, 14, 15, 17, 23, 26, 28. **SUP-10 refutado** y reformulado. Incorporados SUP-29, SUP-30 y SUP-31. SUP-01 ampliado con la limitación medida y la alternativa evaluada. §7 reescrita como tabla que separa medido / adoptado / implementado, tras auditar el código: **SUP-09, 16, 19, 22 y 30 quedan declarados como no implementados**. Añadida advertencia de reproducibilidad de los artefactos de recon 07. |

> **Al modificar un supuesto:** actualizar su campo *Estado*, registrar el resultado numérico obtenido, e incrementar la versión del documento. El campo `assumptions_version` de `metadata.json` debe reflejar la versión vigente al momento de la ejecución.

