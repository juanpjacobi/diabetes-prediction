# 🧪 Clasificación de Diabetes con Random Forest

Este proyecto implementa un flujo completo de Machine Learning para la predicción de niveles de diabetes utilizando Random Forest. Se estructura de forma profesional, modular y preparada para producción.

---

## 📁 Estructura del Proyecto

diabetes_/
├── data/ # Dataset original
├── notebooks/ # Exploraciones y pruebas manuales
├── src/ # Código fuente modular
├── outputs/ # Modelos entrenados, gráficos
├── requirements.txt # Dependencias del proyecto
├── README.md # Descripción general
└── .gitignore # Archivos a excluir del control de versiones



---

## ⚙️ Requisitos

Python 3.9+  
Librerías: ver `requirements.txt`

---

## 🧪 Cómo usar

### 1. Crear entorno virtual

```bash
python -m venv .venv

### para windows

.venv\Scripts\activate

### en mac o linux

source .venv/bin/activate

## Instalar dependencias

pip install -r requirements.txt

## Ejecutar flujo completo desde la consola

python src/main.py

## Alternativamente, podés trabajar desde la notebook:

notebooks/exploracion_modelado.ipynb

## 📊Objetivo del modelo

Clasificar entre:

0: Sin diabetes

1: Prediabetes / Diabetes leve

2: Diabetes avanzada

Maximizar el recall en clases 1 y 2.

Ajustar umbral para evitar falsos negativos.


## Dataset
https://www.kaggle.com/datasets/yasserhessein/multiclass-diabetes-dataset