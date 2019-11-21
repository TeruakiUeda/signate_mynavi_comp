import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from logger import LOGGER
from catboost import Pool
from catboost import CatBoostRegressor


def train_lgbm(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name,
               fold_id, lgb_params, fit_params, model_name, loss_func, rank=False, calc_importances=True):
    train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)
    if X_valid is not None:
        valid = lgb.Dataset(X_valid, y_valid,
                            categorical_feature=categorical_features,
                            feature_name=feature_name)
    evals_result = {}
    if X_valid is not None:
        model = lgb.train(
            lgb_params,
            train,
            valid_sets=[valid],
            valid_names=['valid'],
            evals_result=evals_result,
            **fit_params
        )
    else:
        model = lgb.train(
            lgb_params,
            train,
            evals_result=evals_result,
            **fit_params
        )
    LOGGER.info(f'Best Iteration: {model.best_iteration}')

    # train score
    if X_valid is None:
        y_pred_train = model.predict(X_train, num_iteration=fit_params["num_boost_round"])
        y_pred_train[y_pred_train<0] = 0
        train_loss = loss_func(y_train, y_pred_train)
    else:
        y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
        y_pred_train[y_pred_train < 0] = 0
        train_loss = loss_func(y_train, y_pred_train)

    if X_valid is not None:
        # validation score
        y_pred_valid = model.predict(X_valid)
        y_pred_valid[y_pred_valid < 0] = 0
        valid_loss = loss_func(y_valid, y_pred_valid)
        # save prediction
        #np.save(f'{model_name}_train_fold{fold_id}.npy', y_pred_valid)
    else:
        y_pred_valid = None
        valid_loss = None

    # save model
    """要編集"""
    model.save_model(os.path.join(f'../output/{model_name}', f'{model_name}_fold{fold_id}.txt'))

    if X_test is not None:
        # predict test
        y_pred_test = model.predict(X_test)
        y_pred_test[y_pred_test < 0] = 0
        # save prediction
        #np.save(f'{model_name}_test_fold{fold_id}.npy', y_pred_test)
    else:
        y_pred_test = None

    if calc_importances:
        importances = pd.DataFrame()
        importances['feature'] = feature_name
        importances['gain'] = model.feature_importance(importance_type='gain')
        importances['split'] = model.feature_importance(importance_type='split')
        importances['fold'] = fold_id
    else:
        importances = None

    return y_pred_valid, y_pred_test, train_loss, valid_loss, importances, model.best_iteration


def train_cat(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name,
               fold_id, lgb_params, fit_params, model_name, loss_func, rank=False, calc_importances=True):

    train = Pool(X_train, y_train, cat_features=categorical_features, feature_names=feature_name)
    valid = Pool(X_valid, y_valid, cat_features=categorical_features, feature_names=feature_name)

    evals_result = {}
    model = CatBoostRegressor(verbose=False,random_seed=42, learning_rate=0.1)
    model.fit(train,
              eval_set=valid,  # 検証用データ
              early_stopping_rounds=100,  # 10回以上精度が改善しなければ中止
              use_best_model=True,  # 最も精度が高かったモデルを使用するかの設定
              plot=False,
              verbose=False,)  # 誤差の推移を描画するか否かの設定

    y_pred_train = model.predict(X_train)
    y_pred_train[y_pred_train<0] = 0
    train_loss = loss_func(y_train, y_pred_train)


    if X_valid is not None:
        # validation score
        y_pred_valid = model.predict(X_valid)
        y_pred_valid[y_pred_valid < 0] = 0
        valid_loss = loss_func(y_valid, y_pred_valid)
        # save prediction
        #np.save(f'{model_name}_train_fold{fold_id}.npy', y_pred_valid)
    else:
        y_pred_valid = None
        valid_loss = None

    # save model
    model.save_model(f'{model_name}_fold{fold_id}.txt')

    if X_test is not None:
        # predict test
        y_pred_test = model.predict(X_test)
        y_pred_test[y_pred_test < 0] = 0
        # save prediction
        #np.save(f'{model_name}_test_fold{fold_id}.npy', y_pred_test)
    else:
        y_pred_test = None

    return y_pred_valid, y_pred_test, train_loss, valid_loss
