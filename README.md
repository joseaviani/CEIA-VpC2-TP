# Detección de Objetos Peligrosos en Equipaje mediante Rayos X

## Trabajo Práctico Integrador - Visión por Computadora II

**CEIA - Cohorte 21Co2025**

---

###  Autores

- **José Luis Diaz** - diazjoseluis@gmail.com
- **Ricardo Silvera** - rsilvera@thalu.com.ar  
- **José Aviani** - jose.aviani@gmail.com

---

## Resumen del Proyecto

Este proyecto aborda el problema de **detección automática de objetos peligrosos** en imágenes de equipaje escaneado por rayos X, un desafío crítico para la seguridad en aeropuertos y puntos de control. El objetivo es desarrollar modelos de visión por computadora capaces de identificar objetos potencialmente peligrosos (tijeras, navajas, cuchillos, cutters y navajas multiuso) para asistir al personal de seguridad en la inspección de equipaje.

### Objetivos

1. Implementar y evaluar diferentes arquitecturas de detección de objetos
2. Optimizar los modelos priorizando la métrica **Recall** (sensibilidad) por razones de seguridad
3. Comparar el desempeño entre modelos basados en YOLO y Faster R-CNN
4. Establecer un baseline sólido y documentar mejoras iterativas

---

## Dataset
**Nombre:** X-ray Baggage Detection - Prohibited Items  
**Fuente:** [Roboflow Universe](https://universe.roboflow.com/malek-mhnrl/x-ray-baggage-detection)  
**Licencia:** CC BY 4.0

### Características del Dataset

- **Tipo:** Detección de objetos (Object Detection)
- **Formato:** Imágenes JPG con anotaciones en formato YOLO
- **Clases detectadas:** 5 categorías de objetos peligrosos
  - Clase 0: Tijera
  - Clase 1: Navaja
  - Clase 2: Cuchillo
  - Clase 3: Cutter
  - Clase 4: Navaja multiuso



## Métrica Principal: Recall

### ¿Por Qué Priorizar el Recall?

En un sistema de **seguridad crítica** como la detección de objetos peligrosos en equipaje, la métrica más importante a maximizar es el **Recall (Sensibilidad)**.

#### Análisis de Consecuencias

| Tipo de Error | Significado | Impacto | Severidad |
|---------------|-------------|---------|-----------|
| **Falso Negativo (FN)** | No detectar un objeto peligroso real | **CRÍTICO**: Objeto peligroso pasa desapercibido | **INACEPTABLE** |
| **Falso Positivo (FP)** | Detectar algo como peligroso cuando no lo es | ✓ Manejable: Inspección manual adicional | **TOLERABLE** |

#### Definición del Recall

$$\text{Recall} = \frac{\text{Verdaderos Positivos (TP)}}{\text{Verdaderos Positivos (TP)} + \text{Falsos Negativos (FN)}}$$

El Recall responde a: *"De todos los objetos peligrosos presentes, ¿cuántos detectamos?"*

**Maximizar Recall = Minimizar objetos peligrosos no detectados**

#### Consideración Importante sobre Clasificación

**La clasificación incorrecta NO es crítica en este contexto.** Si el modelo detecta un objeto peligroso pero lo clasifica en la categoría equivocada (por ejemplo, detecta un cuchillo pero lo clasifica como tijera), esto **igualmente disparará el proceso de revisión manual del equipaje**. Lo fundamental es que el sistema **no deje pasar ningún objeto peligroso sin detectar**, independientemente de su clasificación específica.

Por lo tanto:
- **Prioridad máxima:** Alta tasa de detección (Recall alto)
- **Aceptable:** Errores en la clasificación específica del objeto
- **Tolerable:** Falsos positivos que requieren revisión manual
- **Inaceptable:** Falsos negativos (objetos no detectados)

### Métricas Complementarias

Además del Recall, monitoreamos:

- **Precision:** Para evitar saturar el sistema con falsos positivos
- **F2-Score:** Balance que da doble peso al Recall sobre Precision


---

## Modelos Implementados

Este proyecto explora múltiples arquitecturas de detección de objetos, cada una con sus propias características y optimizaciones:

### 1. **YOLOv8 - Baseline** 
- **Arquitectura:** YOLOv8n (nano) pre-entrenado en COCO
- **Enfoque:** Transfer Learning con fine-tuning
- **Configuración inicial:**
  - Resolución: 416×416
  - Épocas: 100
  - Optimizador: SGD con momentum
- **Resultados baseline:** Establecimiento de métricas de referencia

### 2. **YOLOv8 - Mejorado** 
- **Arquitectura:** YOLOv8s (small) pre-entrenado en COCO
- **Mejoras implementadas:**
  - ↑ Resolución: 832×832
  - ↑ Épocas: 150 con early stopping
  - Cosine annealing learning rate + warmup
  - **Oversampling focalizado** en Clase 2 (Cuchillo) - la clase más problemática
- **Resultados destacados:**
  - Recall macro: +4.7% vs baseline
  - Recall Clase 2: +14.4% vs baseline
  - mAP@0.5: 0.898

### 3. **Faster R-CNN (ResNet50-FPN)** 
- **Arquitectura:** Faster R-CNN con backbone ResNet50-FPN
- **Enfoque:** Arquitectura clásica de dos etapas para detección de objetos
- **Características:**
  - Region Proposal Network (RPN) para generación de candidatos
  - Clasificación y regresión de bounding boxes en segunda etapa
  - Pre-entrenamiento en COCO dataset
- **Entrenamiento:** Detalles completos en el notebook correspondiente

---
