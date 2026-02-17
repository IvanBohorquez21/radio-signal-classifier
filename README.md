# 📥〰️📡 Clasificador de Modulaciones de Radio con Deep Learning

Este proyecto utiliza una Red Neuronal Convolucional **(CNN)** desarrollada en PyTorch para identificar y clasificar automáticamente 10 tipos de modulaciones de radio (AMC). Se utiliza el dataset estándar de la industria RML2016.10a, procesando señales en cuadratura (I/Q) para entender el comportamiento de la radiofrecuencia mediante Inteligencia Artificial.

## 🧠 Arquitectura y Entrenamiento

El modelo procesa señales representadas como tensores de entrada de tamaño (2, 128), correspondientes a los componentes en fase (I) y cuadratura (Q).

### Visualización de Señales I/Q
Antes del entrenamiento, el sistema procesa y limpia los datos, permitiendo visualizar la naturaleza de cada modulación:

| 8PSK (Digital) | CPFSK (Digital) |
| --- | --- |





## 🏋️‍♂️🤖 Entrenamiento
Primero se limpian los datos, en este caso al ser señales electricas se puede tomar un criterio de umbral de ruido donde el SNS(dB) sugerido aceptable es un rango de (15dB - 20 dB) para ellos eliminamos todas las frecuencias que estan pordebajo de 20 dB ya que para efectos practicos se tomaran como ruido,en este rango se suele utilizar para las modulaciones de radio frecuencia de baja velocidad.

![ejemplo 1](img/ejemplo_8PSK.png)

![ejemplo 2](img/ejemplo_CPFSK.png)



El modelo fue entrenado durante 60 épocas, logrando una convergencia estable como se muestra en la curva de pérdida:

![curva de aprendizaje](img/curva_aprendizaje_v3.png)


## 📊 Resultados
El rendimiento del modelo se evalúa mediante una matriz de confusión normalizada, que permite identificar la precisión del clasificador para cada tipo de señal, incluso en entornos con ruido.

![Matriz de Confusión](img/matriz_confusion_v3.png)

### Demos de Predicción en Tiempo Real

El script *demo_final.py* permite tomar señales aleatorias y observar la confianza de la IA en su predicción:

![predicción del demo1](img/demo_prediccion_AM-DSB.png)

![predicción del demo2](img/demo_prediccion_AM-SSB.png)

![predicción del demo3](img/demo_prediccion_WBFM.png)

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.13.1
* **Framework:** PyTorch (Deep Learning)
* **Procesamiento:** NumPy, Scikit-learn
* **Visualización:** Matplotlib, Seaborn
* **Dataset:** RadioML 2016.10a

## 🚀 Cómo usar

### 1. Clonar el repositorio

```bash
git clone https://github.com/IvanBohorquez21/radio-signal-classifier
cd radio-signal-classifier

```

### 2. Preparar el entorno e instalar dependencias
 Se recomienda usar un entorno virtual
```bash
python -m venv env_ia
source env_ia/bin/activate  # En Windows: env_ia\Scripts\activate
pip install -r requirements.txt

```

### 3. Flujo de trabajo

1. **Datos:** Descarga el dataset y colócalo en `data/raw/RML2016.10a_dict.pkl`.
2. **Procesamiento:** Ejecuta `notebooks/dataset.ipynb` para generar el archivo optimizado `RML2016_limpio.pt`.
3. **Entrenamiento:** Ejecuta `notebooks/train.ipynb` para entrenar el modelo y guardar los pesos en `models/`.
4. **Prueba:** Corre `src/demo_final.py` para ver la IA en acción.

---

*Desarrollado por [Ivan Bohorquez*]()