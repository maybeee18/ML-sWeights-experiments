import numpy as np
import pandas as pd
import sklearn
from hep_ml import splot
from scipy.special import expit
import catboost

def train_on_sWeights_signal_vs_background_naive(x, sWeights, model):
    full_x = np.concatenate([x, x], axis=0)
    full_weights = np.concatenate([sWeights, 1. - sWeights])
    full_y = np.concatenate([np.ones(x.shape[0]), np.zeros(x.shape[0])])
    model_ = sklearn.clone(model)
    shuffle = np.random.permutation(full_x.shape[0])
    model_.fit(full_x[shuffle], full_y[shuffle], sample_weight=full_weights[shuffle])
    return model_
    

def train_on_sWeights_signal_vs_background_smart(x, sWeights, model):
    """
    Model must be catboost with ConstrainedRegression loss
    """
    if model.get_params()['loss_function'] != 'ConstrainedRegression':
        raise ValueError("Smart training requires catboost with ConstrainedRegression loss")
    model_ = sklearn.clone(model)
    model_.fit(x, sWeights)
    return model_


def train_on_labels_signal_vs_background(x, labels, model):
    model_ = sklearn.clone(model)
    model_.fit(x, labels)
    return model_


def sWeights_to_proba(x, sWeights, model, use_cross_val=False, cv_params={"cv": 4, "n_jobs": 1}):
    if model.get_params()['loss_function'] != 'ConstrainedRegression':
        raise ValueError("Smart training requires catboost with ConstrainedRegression loss")
    if use_cross_val:
        raw_predictions = sklearn.model_selection.cross_val_predict(
            model, x, sWeights, **cv_params)
    else:
        model_ = sklearn.clone(model)
        model_.fit(x, sWeights)
        raw_predictions = model_.predict(x)
    return expit(raw_predictions)


def evaluate_signal_vs_background(model_trainer_labels, x_train, x_test, y_test):
    """
    A highly ugly function tailered for Pool.map(functools.partial
    """
    trained_model = model_trainer_labels[1](x_train, model_trainer_labels[2], model_trainer_labels[0])
    if isinstance(trained_model, catboost.CatBoostClassifier):
        kwargs = {"prediction_type" :"RawFormulaVal"}
    else:
        kwargs = {}
    test_predictions = trained_model.predict(x_test, **kwargs)
    return sklearn.metrics.roc_auc_score(y_test, test_predictions)


def evaluate_regression(args, x_train, y_train, x_test, y_test):
    model_regression, model_weights, weights, use_cv = args
    if model_weights is not None:
        weights = sWeights_to_proba(x_train, weights, model_weights, use_cv)
    model_regression.fit(x_train, y_train, sample_weight=weights)
    test_predictions = model_regression.predict(x_test)
    return sklearn.metrics.mean_squared_error(y_test, test_predictions)

def perdict_raw(model, *args, **kwargs):
    """
    Tries to call model.predict(*args, **kwargs, prediction_type="RawFormulaVal"). If that fail,
    calls model.predict(*args, **kwargs)
    """
    try:
        return model.predict(*args, **kwargs, prediction_type="RawFormulaVal")
    except TypeError:
        return model.predict(*args, **kwargs)