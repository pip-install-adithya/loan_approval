import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

target = 'loan_status'
X_train = train_df.drop(columns=[target, 'id'])
y_train = train_df[target]
X_test = test_df.drop(columns=['id'])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

lgbm = LGBMClassifier(random_state=42)
lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', lgbm)])

param_dist = {
    'classifier__n_estimators': np.arange(100, 500, step=50),
    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'classifier__num_leaves': np.arange(20, 150, step=10),
    'classifier__max_depth': np.arange(3, 10),
    'classifier__min_child_samples': np.arange(10, 100, step=10),
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(estimator=lgbm_pipeline, 
                                   param_distributions=param_dist, 
                                   n_iter=50,
                                   scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=2)

random_search.fit(X_train, y_train)

print("Best Parameters from Random Search:", random_search.best_params_)
print("Best ROC-AUC Score from Random Search:", random_search.best_score_)

y_pred_proba_test = random_search.best_estimator_.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'id': test_df['id'], 'loan_status': y_pred_proba_test})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

lgbm_pipeline.fit(X_train_split, y_train_split, 
                  classifier__eval_set=[(X_val_split, y_val_split)], 
                  classifier__early_stopping_rounds=50,
                  classifier__eval_metric='auc')

y_pred_proba_test_early_stopping = lgbm_pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'id': test_df['id'], 'loan_status': y_pred_proba_test_early_stopping})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created with early stopping.")

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 30,
    'metric': 'auc'
}

cv_results = lgb.cv(params, train_data, num_boost_round=1000, nfold=5, 
                    early_stopping_rounds=50, metrics='auc', seed=42)

print("Best number of boosting rounds:", len(cv_results['auc-mean']))

lgbm = LGBMClassifier(n_estimators=len(cv_results['auc-mean']), **params)
lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])
lgbm_pipeline.fit(X_train, y_train)

y_pred_proba_test_cv = lgbm_pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'id': test_df['id'], 'loan_status': y_pred_proba_test_cv})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created with cross-validation.")

lgbm_pipeline.fit(X_train, y_train)
importance = lgbm_pipeline.named_steps['classifier'].feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df)

top_features = importance_df['Feature'].head(10)
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

lgbm_pipeline_top = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', lgbm)])
lgbm_pipeline_top.fit(X_train_top, y_train)

y_pred_proba_test_top = lgbm_pipeline_top.predict_proba(X_test_top)[:, 1]

submission = pd.DataFrame({'id': test_df['id'], 'loan_status': y_pred_proba_test_top})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created with top features.")
