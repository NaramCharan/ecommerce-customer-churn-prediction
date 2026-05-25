

import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pipelines import Project_pipelines

import models_optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from preprocessing import data_separation
from models_optuna import Optuna_tree_models

data_spliting = data_separation()
X_train, x_test, Y_train, y_test = data_spliting.spliting()
numerical_cols = ['HourSpendOnApp', 'NumberOfDeviceRegistered',  'SatisfactionScore', 'NumberOfAddress', 'Complain','OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       'DaySinceLastOrder']
robust_cols = ['Tenure',  'WarehouseToHome', 'CashbackAmount']
categorical_cols = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode',  'Gender', 'PreferedOrderCat', 'MaritalStatus']

#---------------------Pipelines---------------------
project_pipelines = Project_pipelines()
preprocessing = project_pipelines.pipeline(numerical_cols=numerical_cols, categorical_cols=categorical_cols, robust_cols=robust_cols)

#---------------------LogisticRegression---------------------

logistic_model = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', LogisticRegression(max_iter=5000))
])
logistic_model.fit(X_train, Y_train)
logistic_model_predictions = logistic_model.predict(x_test)
cv_sets = StratifiedKFold(n_splits=10, shuffle=True, random_state=157)
score = cross_val_score(estimator=logistic_model, X=X_train, y=Y_train, cv=cv_sets, n_jobs=-1)

log_acc = np.mean(score)
classification_report_log = classification_report(y_test, logistic_model_predictions, output_dict=True)
log_precision = classification_report_log['1']['precision']
log_recall = classification_report_log['1']['recall']
log_f1 =  classification_report_log['1']['f1-score']

print(f'The accuracy of the logistic regression model is {log_acc}')
print(f'The precision of the logistic regression model is {log_precision}')
print(f'The recall of the logistic regression model is {log_recall}')
print(f'The F1 score of the logistic regression model is {log_f1}')

#---------------------RandomForest---------------------

optuna_tree = Optuna_tree_models(preprocessing=preprocessing, X_train=X_train, Y_train=Y_train)

study_RF = optuna.create_study(direction='maximize')
study_RF.optimize(optuna_tree.objective_RF, n_jobs=-1, n_trials=100)
random_forest_model = Pipeline(steps=[
    ("preprocesing", preprocessing),
    ('model', RandomForestClassifier(**study_RF.best_params, n_jobs=-1, random_state=26, class_weight='balanced'))
])
random_forest_model.fit(X_train, Y_train)
Y_predcitions_RF = random_forest_model.predict(x_test)
RF_acc = accuracy_score(y_test, Y_predcitions_RF)
classification_report_RF = classification_report(y_test, Y_predcitions_RF, output_dict=True)
RF_precision = classification_report_RF['1']['precision']
RF_recall = classification_report_RF['1']['recall']
RF_f1 =  classification_report_RF['1']['f1-score']


print(f'The accuracy of the Random Forest model is {RF_acc}')
print(f'The precision of the  Random Forest  model is {RF_precision}')
print(f'The recall of the Random Forest  model is {RF_recall}')
print(f'The F1 score of the Random Forest  model is {RF_f1}')

#---------------------XGBoost---------------------

study_xg = optuna.create_study(direction='maximize')
study_xg.optimize(func=optuna_tree.objective_xg, n_jobs=-1, n_trials=200)
negative = np.where(Y_train==1, 1, 0)
ratio = (len(negative)-sum(negative))/sum(negative)
xgboost_model = Pipeline(steps=[
    ("preprocesing", preprocessing),
    ('model', XGBClassifier(**study_xg.best_params, n_jobs=-1, random_state=26, scale_pos_weight=ratio))
])
xgboost_model.fit(X_train, Y_train)
Y_predcitions_xg = xgboost_model.predict(x_test)
xg_acc = accuracy_score(y_test, Y_predcitions_xg)
classification_report_xg = classification_report(y_test, Y_predcitions_xg, output_dict=True)
xg_precision = classification_report_xg['1']['precision']
xg_recall = classification_report_xg['1']['recall']
xg_f1 =  classification_report_xg['1']['f1-score']

print(f'The accuracy of the XGBoost model is {xg_acc}')
print(f'The precision of the XGBoost model is {xg_precision}')
print(f'The recall of the XGBoost model is {xg_recall}')
print(f'The F1 score of the XGBoost model is {xg_f1}')
#---------------------Final Evaluationg---------------------

final_metrics_dict = {'Model':['Logistic Regression', 'Random Forest', 'XGBoost'],
                 'Accuracy':[log_acc, RF_acc, xg_acc],
                 'Precision':[log_precision, RF_precision, xg_precision],
                 'Recall':[log_recall, RF_recall, xg_recall],
                 'F1-Score':[log_f1, RF_f1, xg_f1]}
final_metrics = pd.DataFrame(data=final_metrics_dict)
print(final_metrics)