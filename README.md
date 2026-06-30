# Proyecto de Optimizacion: LASSO y Elastic Net

Este repositorio contiene el desarrollo del proyecto de optimizacion aplicado a regresion lineal regularizada. El objetivo es comparar dos modelos, LASSO y Elastic Net, usando un dataset de precios inmobiliarios de Lima Metropolitana.

El trabajo incluye analisis exploratorio, preprocesamiento comun para ambos modelos, implementaciones manuales mediante coordinate descent y validacion contra `scikit-learn`.

## Estructura del proyecto

```text
.
├── data/
│   └── housing_lima_final.xlsx
├── notebooks/
│   ├── 1_analisis_exploratorio.ipynb
│   ├── 2_desarrollo_lasso.ipynb
│   └── 3_desarrollo_elasticnet.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── lasso_manual.py
│   └── elasticnet_manual.py
└── README.md
```

## Requisitos

Se recomienda usar Python 3.10 o superior.

Instalar las dependencias principales:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl jupyter
```

Si se usa un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl jupyter
```

En Windows, la activacion del entorno virtual seria:

```bash
.venv\Scripts\activate
```

## Como ejecutar el proyecto

1. Clonar el repositorio:

```bash
git clone https://github.com/GG-369/Proyecto-Optimizacion-LASSO-ELASTICNET.git
cd Proyecto-Optimizacion-LASSO-ELASTICNET
```

2. Verificar que el archivo de datos este en:

```text
data/housing_lima_final.xlsx
```

3. Abrir Jupyter Notebook:

```bash
jupyter notebook
```

4. Ejecutar los notebooks en este orden:

```text
notebooks/1_analisis_exploratorio.ipynb
notebooks/2_desarrollo_lasso.ipynb
notebooks/3_desarrollo_elasticnet.ipynb
```

Es importante mantener ese orden porque primero se revisa el comportamiento del dataset y luego se entrenan los modelos.

## Que hace cada notebook

### 1. Analisis exploratorio

Revisa la distribucion de precios, la transformacion logaritmica de la variable objetivo y la presencia de correlacion entre predictores. Esta parte justifica el uso de regularizacion en vez de una regresion lineal ordinaria.

### 2. Desarrollo LASSO

Implementa LASSO desde cero usando coordinate descent y soft-thresholding. Tambien realiza busqueda de hiperparametro con validacion cruzada y compara el resultado con `sklearn.linear_model.Lasso`.

### 3. Desarrollo Elastic Net

Implementa Elastic Net desde cero combinando penalizacion L1 y L2. Se prueban distintos valores de lambda y alpha mediante validacion cruzada, y luego se valida el resultado contra `sklearn.linear_model.ElasticNet`.

## Preprocesamiento

El archivo `src/preprocessing.py` centraliza la carga y preparacion de datos. Ambos modelos usan el mismo pipeline:

- carga del archivo Excel;
- definicion del target como `precio_usd_log`;
- separacion train/test 80/20;
- `random_state=42`;
- estandarizacion con `StandardScaler`;
- ajuste del escalador solo con datos de entrenamiento para evitar data leakage.

Esto permite que la comparacion entre LASSO y Elastic Net sea justa.

## Modelos implementados

### LASSO

LASSO resuelve una regresion lineal con penalizacion L1. Su principal ventaja es que puede llevar algunos coeficientes exactamente a cero, por lo que funciona tambien como metodo de seleccion de variables.

Archivo:

```text
src/lasso_manual.py
```

### Elastic Net

Elastic Net combina penalizacion L1 y L2. Mantiene parte de la capacidad de seleccion de LASSO, pero agrega mayor estabilidad cuando existen variables correlacionadas.

Archivo:

```text
src/elasticnet_manual.py
```

## Resultados principales

Con la configuracion usada en los notebooks, ambos modelos obtienen un desempeno muy parecido sobre el conjunto de prueba.

| Modelo | R2 test | RMSE test | MAE test | Variables activas |
|---|---:|---:|---:|---:|
| LASSO | 0.7713 | 0.4009 | 0.2904 | 61 / 70 |
| Elastic Net | 0.7712 | 0.4009 | 0.2903 | 63 / 70 |

La conclusion principal es que Elastic Net no mejora de forma medible el error predictivo frente a LASSO en este dataset. LASSO queda como la opcion mas simple y parsimoniosa, mientras que Elastic Net conserva ligeramente mas variables y puede ser preferible si se prioriza estabilidad ante predictores correlacionados.

## Problemas comunes

### No encuentra el archivo Excel

Verificar que el archivo exista exactamente en:

```text
data/housing_lima_final.xlsx
```

Los notebooks usan rutas relativas desde la carpeta `notebooks`, por eso el dataset debe mantenerse dentro de `data/`.

### Error al leer archivos `.xlsx`

Instalar `openpyxl`:

```bash
pip install openpyxl
```

### Error al importar modulos de `src`

Ejecutar los notebooks desde la carpeta del proyecto. No mover los notebooks fuera de `notebooks/`, porque usan rutas relativas hacia `../src` y `../data`.

## Integrantes

- Gracia Gamboa
- Gabriela Loli
- Mauricio Aguirre
- Raul Solis

## Repositorio

```text
https://github.com/GG-369/Proyecto-Optimizacion-LASSO-ELASTICNET.git
```
