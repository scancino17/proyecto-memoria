import pickle
from time import strftime
import numpy as np
import pandas as pd
from datetime import datetime
import sys

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import RegressorChain

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Variables de Control
SHOULD_SAMPLE = False
SAMPLE_PER = 0.25
DATASET_NAME = 'Yakutia'
N_FOLDS = 10
RAND_STATE = 42
TEST_SIZE = 0.2
REMOVE_COLUMNS = True
N_REPEATS = 3
LOAD_MODELS = False

# Script
args = sys.argv

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

# Modelos 
estimators = {'SGD': RegressorChain(base_estimator=SGDRegressor(max_iter=1000, tol=1e-3, random_state=RAND_STATE), random_state=RAND_STATE),
              'Linear': RegressorChain(base_estimator=LinearRegression(), random_state=RAND_STATE),
              'SVR': RegressorChain(base_estimator=SVR(), random_state=RAND_STATE),
              'RandomForest': RandomForestRegressor(random_state=RAND_STATE),
              'GradientBoosting': RegressorChain(base_estimator=GradientBoostingRegressor(random_state=RAND_STATE), random_state=RAND_STATE),
              'MLP': MLPRegressor(random_state=RAND_STATE, max_iter = 1000)}

# Hiperparametros optimos
opt_estimators = {
    'Alemana': {'SGD': RegressorChain(base_estimator=SGDRegressor(max_iter=1000, random_state=RAND_STATE, learning_rate='adaptive', loss='huber', penalty='l1'), random_state=RAND_STATE),
              'Linear': RegressorChain(base_estimator=LinearRegression(fit_intercept=False), random_state=RAND_STATE),
              'SVR': RegressorChain(base_estimator=SVR(C=1, epsilon=0.1, kernel='poly'), random_state=RAND_STATE),
              'RandomForest': RandomForestRegressor(random_state=RAND_STATE, criterion='squared_error', max_depth=200, n_estimators=400),
              'GradientBoosting': RegressorChain(base_estimator=GradientBoostingRegressor(random_state=RAND_STATE, criterion='friedman_mse', learning_rate=1, loss='squared_error', n_estimators=200), random_state=RAND_STATE),
              'MLP': MLPRegressor(random_state=RAND_STATE, max_iter = 1000, activation='logistic', hidden_layer_sizes=(20,), learning_rate='constant', solver='lbfgs')},
    'Malaga': {'SGD': RegressorChain(base_estimator=SGDRegressor(max_iter=1000, random_state=RAND_STATE, learning_rate='adaptive', loss='huber', penalty='l1'), random_state=RAND_STATE),
              'Linear': RegressorChain(base_estimator=LinearRegression(fit_intercept=False), random_state=RAND_STATE),
              'SVR': RegressorChain(base_estimator=SVR(C=1, epsilon=0.1, kernel='rbf'), random_state=RAND_STATE),
              'RandomForest': RandomForestRegressor(random_state=RAND_STATE, criterion='squared_error', max_depth=200, n_estimators=400),
              'GradientBoosting': RegressorChain(base_estimator=GradientBoostingRegressor(random_state=RAND_STATE, criterion='squared_error', learning_rate=0.1, loss='huber', n_estimators=200), random_state=RAND_STATE),
              'MLP': MLPRegressor(random_state=RAND_STATE, max_iter = 1000, activation='logistic', hidden_layer_sizes=(40,10), learning_rate='constant', solver='adam')},
    'Yakutia': {'SGD': RegressorChain(base_estimator=SGDRegressor(max_iter=1000, random_state=RAND_STATE, learning_rate='adaptive', loss='huber', penalty='l1'), random_state=RAND_STATE),
              'Linear': RegressorChain(base_estimator=LinearRegression(fit_intercept=False), random_state=RAND_STATE),
              'SVR': RegressorChain(base_estimator=SVR(C=1, epsilon=0.1, kernel='poly'), random_state=RAND_STATE),
              'RandomForest': RandomForestRegressor(random_state=RAND_STATE, criterion='squared_error', max_depth=200, n_estimators=400),
              'GradientBoosting': RegressorChain(base_estimator=GradientBoostingRegressor(random_state=RAND_STATE, criterion='friedman_mse', learning_rate=1, loss='squared_error', n_estimators=200), random_state=RAND_STATE),
              'MLP': MLPRegressor(random_state=RAND_STATE, max_iter = 1000, activation='logistic', hidden_layer_sizes=(20,), learning_rate='constant', solver='lbfgs')}
}

# Funciones de ayuda 
def load_pickle(filename: str):
    infile = open(filename, 'rb')
    var = pickle.load(infile, encoding='latin1')
    infile.close()
    return var

def save_to_pickle(filename: str, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def log_msg(msg: str):
    print('[{}] {}'.format(datetime.now().strftime('%H:%M:%S'), msg))


log_msg('Comenzado prueba dataset {}'.format(DATASET_NAME))

# Carga de datos
df = pd.read_csv(DF_PATH, index_col=0)

df = df.drop_duplicates()

# Descripción estadística
df.describe().T.drop(columns=['count']).to_latex('{}_stats.tex'.format(DATASET_NAME))

# Preprocesado de valores faltantes
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

# Modelado
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

X, X_test, y, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND_STATE)

# Hacer kfold, shuffle dataset
cv = KFold(n_splits=N_FOLDS, random_state=RAND_STATE, shuffle=True)

model_scores = {}

log_msg('Comenzando cross validate, hiperparametros predeterminados... ')


if LOAD_MODELS:
    model_scores = load_pickle('{}_Pred_Score.pickle'.format(DATASET_NAME))
else:
    for key, estimator in estimators.items():
        model_scores[key] = cross_validate(estimator, X, y, cv=cv, scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'), n_jobs=-1, return_estimator=True)
    save_to_pickle('{}_Pred_Score.pickle'.format(DATASET_NAME), model_scores)

log_msg('Cross validate terminado.')

# Puntaje medio obtenido en cross val, hiperparametros predeterminados
model_scores_df =  pd.DataFrame.from_dict(model_scores).drop('estimator')
model_scores_df = model_scores_df.apply(lambda x: x.apply(lambda y: np.mean(y)), axis=1)
model_scores_df

filter = {}
cv_scores = {}
test_scores = { 'MAE': {}, 'R2': {}, 'MSE': {}}
test_keys = []
model_keys = list(model_scores.keys())

for key, item in model_scores.items():
    item_keys = [x for x in item.keys() if 'test' in x]
    test_keys = item_keys
    filter[key] = {}
    for ikey in item_keys:
        filter[key][ikey] = item[ikey]

for test in test_keys:
    cv_scores[test] = {}
    for model in model_keys:
        cv_scores[test][model] = filter[model][test]

log_msg('Evaluando test set, hiperparametros predeterminados...')

for key, item in model_scores.items():
    test_scores['MAE'][key] = list()
    test_scores['R2'][key] = list()
    test_scores['MSE'][key] = list()
    for estimator in model_scores[key]['estimator']:
        y_pred = estimator.predict(X_test)
        test_scores['MAE'][key].append(mean_absolute_error(y_test, y_pred))
        test_scores['MSE'][key].append(mean_squared_error(y_test, y_pred))
        test_scores['R2'][key].append(r2_score(y_test, y_pred))

log_msg('Test set evaluado.')

test_label_dict = {'test_neg_mean_squared_error': 'MSE',
                   'test_neg_mean_absolute_error': 'MAE',
                   'test_r2': 'R2'}

for test, results in cv_scores.items():
    fig_df = pd.DataFrame.from_dict(results)

    if 'neg' in test:
        fig_df = fig_df.apply(lambda x: x.apply(lambda y: -y))

    fig_df.to_csv('./{}_{}_cross_val_pred_hyper.csv'.format(DATASET_NAME, test_label_dict[test]))

# Tener cuidado con signo de pruebas
for test in ['MAE', 'MSE', 'R2']:
    test_scores_df = pd.DataFrame.from_dict(test_scores[test])
    test_scores_df.to_csv('./{}_{}_test_pred_hyper.csv'.format(DATASET_NAME, test))

# Hiperparametros optimos
model_scores = {}

log_msg('Comenzando cross validate, hiperparametros optimos... ')


if LOAD_MODELS:
    model_scores = load_pickle('{}_Opt_Score.pickle'.format(DATASET_NAME))
else:
    for key, estimator in opt_estimators[DATASET_NAME].items():
        model_scores[key] = cross_validate(estimator, X, y, cv=cv, scoring=('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'), n_jobs=-1, return_estimator=True)
    save_to_pickle('{}_Opt_Score.pickle'.format(DATASET_NAME), model_scores)


log_msg('Cross validate terminado.')

model_scores_df =  pd.DataFrame.from_dict(model_scores).drop('estimator')
model_scores_df = model_scores_df.apply(lambda x: x.apply(lambda y: np.mean(y)), axis=1)

filter = {}
cv_scores = {}
test_keys = []
model_keys = list(model_scores.keys())

for key, item in model_scores.items():
    item_keys = [x for x in item.keys() if 'test' in x]
    test_keys = item_keys
    filter[key] = {}
    for ikey in item_keys:
        filter[key][ikey] = item[ikey]

for test in test_keys:
    cv_scores[test] = {}
    for model in model_keys:
        cv_scores[test][model] = filter[model][test]

test_scores = { 'MAE': {}, 'R2': {}, 'MSE': {}}

log_msg('Evaluando test set, hiperparametros optimos...')

for key, item in model_scores.items():
    test_scores['MAE'][key] = list()
    test_scores['R2'][key] = list()
    test_scores['MSE'][key] = list()
    for estimator in model_scores[key]['estimator']:
        y_pred = estimator.predict(X_test)
        test_scores['MAE'][key].append(mean_absolute_error(y_test, y_pred))
        test_scores['MSE'][key].append(mean_squared_error(y_test, y_pred))
        test_scores['R2'][key].append(r2_score(y_test, y_pred))

log_msg('Test set evaluado.')

for test, results in cv_scores.items():
    fig_df = pd.DataFrame.from_dict(results)

    if 'neg' in test:
        fig_df = fig_df.apply(lambda x: x.apply(lambda y: -y))

    fig_df.to_csv('./{}_{}_cross_val_opt_hyper.csv'.format(DATASET_NAME, test_label_dict[test]))


# Tener cuidado con signo de pruebas
for test in ['MAE', 'MSE', 'R2']:
    test_scores_df = pd.DataFrame.from_dict(test_scores[test])
    test_scores_df.to_csv('./{}_{}_test_opt_hyper.csv'.format(DATASET_NAME, test))

log_msg('Prueba en dataset {} terminada.'.format(DATASET_NAME))

log_msg('Imprimiendo relevancia de atributos en dataset {}'.format(DATASET_NAME))

# Etiquetas de interppretabilidad
X_cols = list(dataset.drop(columns=['Date', 'renewable', 'cost', 'losses']).drop(columns=[x for x in dataset.columns if 'terminal' in x]).columns)
X_cols.append('losses')
X_cols.append('renewable')
col_length = len(X_cols)

# Caso Linear, SVR
for model in ['Linear', 'SVR', 'SGD']:
    estimator = model_scores['Linear']['estimator'][0]
    estimator.estimators_[0].coef_

    # Agregar valores cero en filas donde no hay valores
    pad_coefs = list()
    for estimator in estimator.estimators_:
        pad_coefs.append(np.pad(estimator.coef_, (0, col_length - len(estimator.coef_)), mode='constant', constant_values=0))

    coefs_df = pd.DataFrame(pad_coefs, columns=X_cols).T.set_axis(['losses', 'renewable', 'cost'], axis=1, inplace=False)
    coefs_df.to_latex('{}_coefs_opt_{}.tex'.format(DATASET_NAME, model))
    coefs_df = coefs_df.apply(lambda x: np.log10(np.maximum(np.abs(x), 1)))
    coefs_df.to_latex('{}_coefs_opt_log_{}.tex'.format(DATASET_NAME, model))

# Caso GradientBoosting
estimator = model_scores['GradientBoosting']['estimator'][0]

pad_coefs = list()
for estimator in estimator.estimators_:
    pad_coefs.append(np.pad(estimator.feature_importances_, (0, col_length - len(estimator.feature_importances_)), mode='constant', constant_values=0))

coefs_df = pd.DataFrame(pad_coefs, columns=X_cols).T.set_axis(['losses', 'renewable', 'cost'], axis=1, inplace=False)
coefs_df.to_latex('{}_relevance_opt_GradientBoosting.tex'.format(DATASET_NAME))

coefs_df = coefs_df.apply(lambda x: np.log10(np.abs(x)))
coefs_df.to_latex('{}_relevance_log_opt_GradientBoosting.tex'.format(DATASET_NAME))

# Caso RandomForest
X_cols = list(dataset.drop(columns=['Date', 'renewable', 'cost', 'losses']).drop(columns=[x for x in dataset.columns if 'terminal' in x]).columns)
rforest_estimator = model_scores['RandomForest']['estimator'][0]
forest_importance_df = pd.DataFrame([rforest_estimator.feature_importances_], columns=X_cols).T
forest_importance_df.to_latex('{}_relevance_opt_RandomForest.tex'.format(DATASET_NAME))
forest_importance_magnitude_df = forest_importance_df.apply(lambda x: np.log10(np.abs(x)))
forest_importance_df.to_latex('{}_relevance_opt_log_RandomForest.tex'.format(DATASET_NAME))




