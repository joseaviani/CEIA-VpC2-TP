

## Objetivo operativo

El sistema toma una decisión a nivel imagen: **abrir el equipaje si se detecta al menos un objeto peligroso**. La **prioridad 1** es minimizar el riesgo, reduciendo al máximo los **falsos negativos** (casos donde hay un objeto peligroso y el modelo lo deja como *background* / no lo detecta). La **prioridad 2** es controlar demoras y costos operativos, reduciendo los **falsos positivos** (casos donde el modelo detecta un objeto peligroso que en realidad no está). Los errores de “confusión” entre objetos peligrosos (p.ej., detectar *cutter* cuando era *cuchillo*) **no son críticos**, porque igualmente justifican la apertura del equipaje.

## Arquitectura utilizada

Se utiliza **Faster R-CNN con backbone ResNet50-FPN** (Torchvision), inicializado con pesos **preentrenados en COCO**, por ser un detector de dos etapas con buen desempeño en precisión/localización y buen comportamiento con objetos relativamente pequeños o difíciles. Además, permite ajustar de forma controlada el **operating point** (umbral de score, NMS y máximo de detecciones) para alinear el sistema con el objetivo operativo. En entrenamientos posteriores se exploran mejoras como **data augmentation “segura” para rayos X** y **ajuste de anchors del RPN** para objetos alargados (cuchillo/cutter).

## Organización de entrenamientos y resultados

Los detalles completos de cada entrenamiento se documentan en su **notebook** correspondiente: al **inicio** se describe la configuración y cambios aplicados (respecto al entrenamiento anterior) y, al **final**, se incluye el análisis de métricas principales, una comparación contra el entrenamiento previo y los **próximos pasos** propuestos.