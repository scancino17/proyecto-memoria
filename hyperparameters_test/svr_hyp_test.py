import numpy as np
import pandas as pd
import sys

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import RegressorChain

RAND_STATE = 42

def load_df(filename: str):
    return pd.read_csv(filename)

args = sys.argv

if len(args) < 3 or len(args) % 2 != 1:
    print('Modo de uso:\nsvr_hyp_test.py [nombre_dataset] [path_dataframe.csv]')
    print('Parámetros adicionales:')
    print('  "--sample": Porcentaje de dataset a usar. Valor entre 0.0 y 1.0. Por defecto: 1.0')
    print('  "--test_size": Porcentaje de dataset a usar como conjunto de prueba. Valor entre 0.0 y 1.0. Por defecto: 0.2')
    print('  "--n_folds": Cantidad de folds para validación cruzada. Valor entero positivo. Por defecto: 10')
    print('  "--n_repeats": Cantidad de repeticiones para K fold. Valor entreo positivo. Por defecto: 3')
    print('  "--max_iter": Cantidad máxima de iteraciones para el entrenamiento de cada modelo. Por defecto 10000')
    sys.exit(0)

valid_args = ['--sample', '--test_size', '--n_folds', '--n_repeats',  '--max_iter']

args_dict_index = dict()
for arg in args:
    if arg in valid_args:
        args_dict_index[arg] = args.index(arg)

arg_values = dict()
for arg, index in args_dict_index.items():
    arg_values[arg] = args[index + 1]

DATASET_NAME = args[1]
DF_PATH = args[2]

SHOULD_SAMPLE = False
SAMPLE_PER = 1.0

TEST_SIZE = 0.2
N_FOLDS = 10
N_REPEATS = 3
MAX_ITER = 10000

if '--sample' in arg_values.keys():
    SHOULD_SAMPLE = True
    SAMPLE_PER = float(arg_values['--sample'])

if '--test_size' in arg_values.keys():
    TEST_SIZE = float(arg_values['--test_size'])

if '--n_folds' in arg_values.keys():
    N_FOLDS = int(arg_values['--n_folds'])

if '--n_repeats' in arg_values.keys():
    N_REPEATS = int(arg_values['--n_repeats'])

if '--max_iter' in arg_values.keys():
    MAX_ITER = int(arg_values['--max_iter'])

ESTIMATOR_NAME = 'SVR'
ESTIMATOR = RegressorChain(base_estimator=SVR(max_iter=MAX_ITER), random_state=RAND_STATE)
HYPER_SPACE = {
        'base_estimator__kernels': ['linear', 'poly', 'rbf', 'sigmoid'],
        'base_estimator__C': [1, 10, 100],
        'base_estimator__epsilon': [0.1, 1, 10]
    }

df = load_df(DF_PATH)
df = df.drop_duplicates()
df = df.replace(np.NaN, 0)

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
df.shape

if SHOULD_SAMPLE:
    dataset = df.sample(frac=SAMPLE_PER, random_state = RAND_STATE)
else:
    dataset = df

# Shuffle dataset
dataset = dataset.sample(frac=1, random_state=RAND_STATE).reset_index(drop=True)

X = dataset.drop(columns=['Date', 'renewable', 'cost', 'losses']).drop(columns=[x for x in dataset.columns if 'terminal' in x])
y = dataset[['losses', 'renewable', 'cost']]

scaler = StandardScaler()
X = scaler.fit_transform(X)

print('Evaluando hiperparámetros para {}: {} filas, {} columnas'.format(ESTIMATOR_NAME, X.shape[0], X.shape[1]))

cv = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RAND_STATE)
X = dataset.drop(columns=['Date', 'renewable', 'cost', 'losses']).drop(columns=[x for x in dataset.columns if 'terminal' in x])
y = dataset[['losses', 'renewable', 'cost']]

search = GridSearchCV(ESTIMATOR, HYPER_SPACE, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
result = search.fit(X, y)
results_df = pd.DataFrame.from_dict(result.cv_results_)
print(result.best_score_)
results_df.to_csv('{}_{}_HyperScores.csv'.format(DATASET_NAME, ESTIMATOR_NAME), sep=';')
