import numpy as np 
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import itertools
import logging

"""
mlp_tools.py

Questo modulo contiene la classe EncoderMLP e le funzioni per eseguire la nested cross-validation.
"""

class EncoderMLP:
    """
    Una classe per un Multi-Layer Perceptron (MLP) Encoder.

    Attributi
    ----------
    model : MLPRegressor
        Il modello MLP da utilizzare per l'addestramento e la predizione.

    Metodi
    -------
    fit(X)
        Addestra l'MLPRegressor con i dati forniti.

    transform(X)
        Ottiene le feature ridotte usando l'output del layer nascosto.

    fit_transform(X)
        Addestra il modello e trasforma i dati in una sola chiamata.
    
    """

    def __init__(self, hidden_layer_size=60, max_iter=50, random_state=None, solver='sgd'):
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.model = MLPRegressor(hidden_layer_sizes=(self.hidden_layer_size,),
                                  max_iter=self.max_iter,
                                  random_state=self.random_state,
                                  solver=self.solver)
        
        self.activation_map = {
            'identity': lambda x: x,
            'logistic': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x)
        }

    def fit(self, X):
        self.model.fit(X, X)

    def transform(self, X):
        hidden_layer_output = np.dot(X, self.model.coefs_[0]) + self.model.intercepts_[0]
        activation_func = self.activation_map[self.model.activation]
        hidden_layer_output = activation_func(hidden_layer_output)
        return hidden_layer_output

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def nested_cv_svm(X, random_seed, decomposition, n_outer_folds=7, n_inner_folds=5, n_components=65, mod_selection_score=accuracy_score, positive_class=0):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('../logs/svm_nested_cv.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger()

    logger.handlers = []
    logger.addHandler(file_handler)

    kernels = ['linear', 'rbf']
    gammas = np.logspace(-3, 3, 7)
    nus = np.linspace(0.01, 0.50, 25) 
    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    
    best_params = {'kernel': '', 'gamma': 0, 'nu': 0, 'components': 0, 'scaler': ''}
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    best_overall_accuracy = 0
    best_overall_params = {'kernel': '', 'gamma': 0, 'nu': 0, 'components': 0, 'scaler': ''}

    y = X['Mezzo'].values
    y = np.array([0. if x == positive_class else 1. for x in y], dtype=float)
    
    X = X.drop(columns='Mezzo').values

    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_seed)
        
    for outer_cv_number, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_seed)

        best_score = 0
        best_scaler = None
        best_encoder = None
        for inner_cv_number, (trainval_idx, valid_idx) in enumerate(inner_cv.split(X_train, y_train)):
            for kernel in kernels:
                param_combinations = itertools.product([kernel], gammas, nus, scalers)
                
                if kernel == 'linear':
                    param_combinations = itertools.product(['linear'], [0.001], nus, scalers)
                elif kernel == 'rbf':
                    param_combinations = itertools.product(['rbf'], gammas, nus, scalers)

                for params in param_combinations:
                    scaler = params[3]
                    X_trainval, X_valid = X_train[trainval_idx], X_train[valid_idx]
                    y_trainval, y_valid = y_train[trainval_idx], y_train[valid_idx]

                    X_trainval_scaled = scaler.fit_transform(X_trainval)
                    X_valid_scaled = scaler.transform(X_valid)

                    encoder = decomposition(hidden_layer_size=n_components, max_iter=50, random_state=random_seed, solver='sgd')
                    encoder.fit(X_train)
                    
                    X_trainval_reduced = encoder.transform(X_trainval_scaled)
                    X_valid_reduced = encoder.transform(X_valid_scaled)
                        
                    idxs_neg = np.where(y_trainval == 1)[0]
                
                    X_trainval_reduced = np.delete(X_trainval_reduced, idxs_neg, axis=0)
                    y_trainval = np.delete(y_trainval, idxs_neg)

                    clf = OneClassSVM(kernel=params[0], gamma=params[1], nu=params[2])
                        
                    clf.fit(X_trainval_reduced)
                        
                    pred_values = clf.predict(X_valid_reduced)
                    true_values = [1 if y == 0 else -1 for y in y_valid]
                        
                    score = mod_selection_score(true_values, pred_values)
                    curr_params = {
                            'kernel': params[0],
                            'gamma': params[1],
                            'nu': params[2],
                            'components': n_components,
                            'scaler': params[3]
                    }

                    logging.info(f"inner cv number: {inner_cv_number}, {mod_selection_score.__name__}: {score}, with params: {curr_params}")
                            
                    if score > best_score:
                        best_score = score
                        best_encoder = encoder
                        best_scaler = scaler
                        best_params = curr_params

        idxs_neg = np.where(y_train == 1)[0]
        X_train = np.delete(X_train, idxs_neg, axis=0)
        y_train = np.delete(y_train, idxs_neg)

        X_train_scaled = best_scaler.fit_transform(X_train)
        X_test_scaled = best_scaler.transform(X_test)

        X_train_reduced = best_encoder.transform(X_train_scaled)
        X_test_reduced = best_encoder.transform(X_test_scaled)

        clf = OneClassSVM(kernel=best_params['kernel'], gamma=best_params['gamma'], nu=best_params['nu'])
        clf.fit(X_train_reduced)

        pred_values = clf.predict(X_test_reduced)
        true_values = [1 if y == 0 else -1 for y in y_test]

        accuracy = accuracy_score(true_values, pred_values)
        precision = precision_score(true_values, pred_values, zero_division=0.0)
        recall = recall_score(true_values, pred_values)
        f1 = f1_score(true_values, pred_values)

        if accuracy > best_overall_accuracy:
            best_overall_accuracy = accuracy
            best_overall_params = best_params

        logging.info(f"outer cv number: {outer_cv_number}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} with params: {best_params}")

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    logging.info(f"algorithm: OneClassSVM, 
                 positive class: {positive_class},
                 bestkernel: {best_overall_params['kernel']}, 
                 best gamma: {best_overall_params['gamma']}, 
                 best nu: {best_overall_params['nu']},
                 n_components: {n_components},
                 best scaler: {best_overall_params['components']},
                 score used for model selection: {mod_selection_score.__name__},
                 accuracy mean: {np.mean(accuracy_scores) * 100},
                 accuracy std: {np.std(accuracy_scores) * 100},
                 precision mean: {np.mean(precision_scores) * 100},
                 precision std: {np.std(precision_scores) * 100},
                 recall mean: {np.mean(recall_scores) * 100},
                 recal std: {np.std(recall_scores) * 100},
                 f1 mean: {np.mean(f1_scores) * 100},
                 f1 std: {np.std(f1_scores) * 100},
                 best overall accuracy: {best_overall_accuracy * 100}")
    return 