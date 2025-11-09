# Clasificación de Ingresos — Adult Income (Random Forest)

## Descripción

Este proyecto implementa un modelo de **Random Forest Classifier** para predecir si una persona gana más de **50K USD al año**, utilizando el conjunto de datos **Adult (UCI Machine Learning Repository)**.  
El objetivo es aplicar un flujo completo de Machine Learning: desde la exploración y limpieza de datos hasta la evaluación del modelo y su interpretación.  
El análisis incluye ingeniería de características, codificación de variables categóricas, ajuste de hiperparámetros y validación del rendimiento.

## Estructura del repositorio

- **`.gitignore`**: Archivo para especificar archivos y carpetas que Git debe ignorar (por ejemplo: datos temporales, entornos virtuales o archivos de IDE).  
- **`environment.yml`**: Archivo de configuración para crear un entorno Conda con todas las dependencias necesarias (Python, bibliotecas como Pandas, Seaborn, etc.).  
- **`README.md`**: Este archivo, con la descripción del proyecto, instrucciones de instalación y uso.   
- **`Data/`**: Carpeta opcional para almacenar el dataset si se desea trabajar de forma local.  
  - En este proyecto, el dataset se carga directamente desde la fuente oficial del **[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)** mediante la siguiente instrucción en el notebook:
    ```python
    df = pd.read_csv('https://archive.ics.uci.edu/static/public/20/data.csv', sep=',')
    ```  
- **`Notebooks/`**: Carpeta con los notebooks de análisis.  
    - `adult income rf model.ipynb`: Notebook principal (EDA, preprocesado, modelado, evaluación e interpretabilidad).  
    - `pandas-missing-extension.ipynb`: Extensión/utility notebook que registra métodos convenientes para inspección y visualización de valores faltantes (accesor `.missing` para DataFrame).

## Instalación y ejecución de repositorio

### Instalación entorno virtual

Con conda (Recomendado):
```bash
conda env create -f environment.yml
conda activate adult-income-rf
```

### Clonación del repositorio

1. Clona el repositorio:
```bash
git clone https://github.com/AlejandroSegura24/Vehicle-Data-Exploration-Project.git
cd Vehicle-Data-Exploration-Project
```

2. Abre el notebook `adult_income_rf_model.ipynb` en Jupyter Notebook o VS Code.
3. Ejecuta las celdas en orden.
4. Revisa las métricas (Accuracy, F1-score, AUC-ROC) y visualizaciones generadas en el notebook.

## Resultados

| Métrica | Entrenamiento | Prueba |
|----------|---------------|--------|
| **Accuracy** | 0.9999 | 0.8589 |
| **AUC-ROC** | — | 0.91 |
| **F1-score (<=50K / >50K)** | — | 0.91 / 0.68 |

El modelo muestra buena capacidad predictiva, con leve sobreajuste y rendimiento menor en la clase >50K.

## Tecnologías usadas

- Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.  
- Entorno: Conda / Jupyter Notebook.  
- Control de versiones: Git y GitHub.

## Flujo del proyecto

1. Carga y limpieza de datos.  
2. Análisis exploratorio (EDA).  
3. Ingeniería de características.  
4. Modelado con Random Forest.  
5. Evaluación e interpretación (métricas y SHAP).

## Conclusiones

- El modelo clasifica correctamente la mayoría de los casos.  
- Ligero sobreajuste en entrenamiento.  
- Variables clave: nivel educativo, horas trabajadas y ocupación.  
- El pipeline es ampliable y reproducible.

## Próximos pasos

- Probar modelos alternativos (XGBoost, LightGBM).  
- Aplicar SMOTE para balancear clases.
- Eliminación de columnas innecesarias para optimizar el modelo.

## Autor

- **Autor**: David Alejandro Segura  
- **Proyecto**: Proyecto académico y práctico sobre análisis de datos y modelado predictivo con Python.  
- **Contacto**: [davidalejandrocmbs@gmail.com](mailto:davidalejandrocmbs@gmail.com)