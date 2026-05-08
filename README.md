# Implementación de un modelo de regresión en Keras

Este repositorio contiene un cuaderno de Jupyter donde se implementan y comparan dos modelos de **redes neuronales para regresión** utilizando **Keras/TensorFlow**. Se trabaja con el dataset **California Housing**, un conjunto de datos clásico para predecir el valor medio de las viviendas.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FribBIG6A1NcR4M7NVmsuBdIKxHaGl_U?usp=sharing)

##  Dataset

Se utiliza **California Housing**, disponible en `sklearn.datasets`. Contiene 20,640 muestras con 8 características numéricas (ingresos, antigüedad, habitaciones, etc.) y una variable objetivo: el valor medio de la casa (`MedHouseVal`).

## Flujo de trabajo

1. **Carga y exploración de datos**  
   - Lectura del dataset con `fetch_california_housing(as_frame=True)`.  
   - Análisis descriptivo: `info()`, `head()`, `tail()`, `describe()`.  
   - Histogramas de todas las variables.

2. **Preprocesamiento**  
   - División entrenamiento/prueba (90% / 10%) con `train_test_split`.  
   - Normalización de las características con `StandardScaler` (ajustado solo al conjunto de entrenamiento).

3. **Modelo 1 – Red sencilla (dos capas ocultas)**  
   - Arquitectura: `8 → 64 → 32 → 1` con activación `relu`, inicializador `he_normal`.  
   - Optimización: `Adam(lr=0.01)`, función de pérdida `mean_squared_error`, métrica `mean_absolute_error`.  
   - Entrenamiento durante 100 épocas.  
   - Evaluación y gráfica de evolución de la pérdida.

4. **Modelo 2 – Red más profunda (tres capas ocultas)**  
   - Arquitectura: `8 → 128 → 64 → 32 → 1` con activación `tanh`, inicializador `he_uniform`.  
   - Optimización: `SGD(lr=0.01, momentum=0.9)`.  
   - Entrenamiento durante 100 épocas.  
   - Evaluación, gráfica y predicción de ejemplo.

## Resultados

| Modelo | Pérdida (test MSE) | MAE (test) |
|--------|---------------------|------------|
| Red 2 capas (Adam, ReLU) | 0.5100 | 0.5769 |
| Red 3 capas (SGD, tanh)  | **0.2143** | **0.3199** |

La red más profunda con tangente hiperbólica y SGD+momentum logra un rendimiento superior, reduciendo tanto el error cuadrático medio como el error absoluto medio.

##  Requisitos

- Python 3.7+
- Librerías principales:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow` (2.x)

Instalación rápida:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
