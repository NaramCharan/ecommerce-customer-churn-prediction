from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder

class Project_pipelines:
    def __init__(self):
        pass

    def pipeline(self, numerical_cols, categorical_cols, robust_cols):
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ("scaler", StandardScaler())
        ])
        robust_pipeline = Pipeline(steps=[
            ('scaler', SimpleImputer(strategy='median')),
            ('robust_scaler', RobustScaler())
        ])
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ('onehotencoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'))

        ])
        preprocessing = ColumnTransformer(transformers=[
            ('numerical_pipeline', numerical_pipeline, numerical_cols),
            ('robust_pipeline', robust_pipeline, robust_cols),
            ('categorical_pipeline', categorical_pipeline, categorical_cols),
        ], remainder='passthrough', n_jobs=-1)

        return preprocessing




