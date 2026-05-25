
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

class Optuna_tree_models:
    def __init__(self, preprocessing, X_train, Y_train):
        self.preprocessing = preprocessing
        self.X_train = X_train
        self.Y_train = Y_train

    def objective_RF(self, trail):
        n_estimators = trail.suggest_int('n_estimators', 100, 1000)
        max_depth = trail.suggest_int('max_depth', 5, 30)
        min_samples_split = trail.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trail.suggest_int('min_samples_leaf', 1, 20)
        max_features = trail.suggest_categorical('max_features', ['log2', 'sqrt'])

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       n_jobs=-1)
        pipeline = Pipeline(steps=[
            ('preprocessing', self.preprocessing),
            ('model', model)
        ])

        cv_sets = StratifiedKFold(n_splits=10, shuffle=True, random_state=157)
        score = cross_val_score(estimator=pipeline, X=self.X_train, y=self.Y_train, cv=cv_sets, n_jobs=-1, scoring='recall')
        return np.mean(score)

    def objective_xg(self, trail):
        n_estimators = trail.suggest_int('n_estimators', 100, 1000)
        max_depth = trail.suggest_int('max_depth', 5, 30)
        learning_rate = trail.suggest_float('learning_rate', 0.01, 0.1)
        subsample = trail.suggest_float('subsample', 0.7, 0.9)

        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                              subsample=subsample, n_jobs=-1)
        pipeline = Pipeline(steps=[
            ("preprocessing", self.preprocessing),
            ('model', model)
        ])
        cv_sets = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
        score = cross_val_score(estimator=pipeline, X=self.X_train, y=self.Y_train, cv=cv_sets, n_jobs=-1, scoring='recall')
        return np.mean(score)
