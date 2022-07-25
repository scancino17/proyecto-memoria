from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from time import strftime
from datetime import datetime
import sys

args = sys.argv

RAND_STATE = 42
TEST_SIZE = 0.2

if len(args) < 3 or len(args) % 2 != 1:
    print('Modo de uso:\nAlgorithmPerformance.py [nombre_dataset] [path_dataframe.csv]')
    print('Parámetros adicionales:')
    print('  "--sample": Porcentaje de dataset a usar. Valor entre 0.0 y 1.0. Por defecto: 1.0')
    print('  "--test_size": Porcentaje de dataset a usar como conjunto de prueba. Valor entre 0.0 y 1.0. Por defecto: 0.2')
    print('  "--n_folds": Cantidad de folds para validación cruzada. Valor entero positivo. Por defecto: 10')
    print('  "--n_repeats": Cantidad de repeticiones para K fold. Valor entreo positivo. Por defecto: 3')
    print('  "--max_iter": Cantidad máxima de iteraciones para el entrenamiento de cada modelo. Por defecto 10000')
    print('  "--load_models": Cargar últimos modelos generados. Por defecto False')
    sys.exit(0)

valid_args = ['--sample', '--test_size', '--n_folds', '--n_repeats',  '--max_iter', '--load_models']

args_dict_index = dict()
for arg in args:
    if arg in valid_args:
        args_dict_index[arg] = args.index(arg)

arg_values = dict()
for arg, index in args_dict_index.items():
    arg_values[arg] = args[index + 1]

DATASET_NAME = args[1]
DF_PATH = args[2]

if '--sample' in arg_values.keys():
    SHOULD_SAMPLE = True
    SAMPLE_PER = float(arg_values['--sample'])

if '--test_size' in arg_values.keys():
    TEST_SIZE = float(arg_values['--test_size'])

if '--n_folds' in arg_values.keys():
    N_FOLDS = int(arg_values['--n_folds'])

if '--n_repeats' in arg_values.keys():
    N_REPEATS = int(arg_values['--n_repeats'])

if '--load_models' in arg_values.keys():
    LOAD_MODELS = True

# Funciones de ayuda 
def log_msg(msg: str):
    print('[{}] {}'.format(datetime.now().strftime('%H:%M:%S'), msg))

log_msg('Comenzado prueba dataset {}'.format(DATASET_NAME))

df = pd.read_csv(DF_PATH, index_col=0)
df = df.drop_duplicates()
df = df.replace(np.NaN, 0)


# Eliminación de variables con baja varianza
selector = VarianceThreshold()
columns = df.columns.drop(['Date', 'renewable', 'cost', 'losses']).drop([x for x in df.columns if 'terminal' in x])
selector.fit(df.drop(columns=['Date', 'renewable', 'cost', 'losses']).drop(columns=[x for x in df.columns if 'terminal' in x]))

selected_columns = list()
removed_columns = list()

for column, selected in zip(columns, selector.get_support()):
    if selected:
        selected_columns.append(column)
    else:
        removed_columns.append(column)

df = df.drop(columns=removed_columns)
log_msg('Columnas removidas por baja varianza:\n{}'.format(removed_columns))

dataset = df.sample(frac=1, random_state=RAND_STATE).reset_index(drop=True)

X = dataset.drop(columns=['Date', 'renewable', 'cost', 'losses']).drop(columns=[x for x in dataset.columns if 'terminal' in x])
y = dataset[['losses', 'renewable', 'cost']]

x_cols = X.columns
X, X_test, y, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND_STATE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)


# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels= x_cols, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap='vlag')
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])

# for i in range(len(dendro["leaves"])):
#     for j in range(len(dendro["leaves"])):
#         text = ax2.text(j, i, np.round(corr[i, j]),
#                        ha="center", va="center", fontsize='xx-small')

fig.tight_layout()

fig.savefig('./figures/{}_correlation_dendogram.png'.format(DATASET_NAME))
