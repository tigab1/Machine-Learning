#!/usr/bin/env python
# coding: utf-8


# Import packages
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, learning_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import mutual_info_regression
from yellowbrick.model_selection import ValidationCurve
import warnings


plt.style.use('ggplot')
warnings.simplefilter('ignore')


t0 = time.time()


# CONSTANTS
DATA_FOLDER = 'data/'
GRAPH_FOLDER = 'graphs/'
RANDOM_STATE = 0
TEST_SIZE = 0.2
K_FOLD_SETTINGS = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# #### Score function


# Define custom score function
def score_f1(y_true, y_pred, threshold):
    return f1_score(y_true=y_true>threshold, y_pred=y_pred>threshold)

def score_regression(y_true, y_pred):
    scores = [score_f1(y_true, y_pred, th) for th in [500, 1400, 5000, 10000]]
    return np.mean(scores)

# [See](https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn)
reg_scorer = make_scorer(score_regression, greater_is_better=True)


# #### Import the data

# Import the data
X1 = pd.read_csv(DATA_FOLDER +  'X1.csv')
Y1 = pd.read_csv(DATA_FOLDER + 'Y1.csv', header=None, names=['shares'])

# Make a copy of data to work with
df = X1.copy()
target = Y1.copy()


# #### Split the data into train/test


### Split the data into train/test --> 80%/20%
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_test:\t', X_test.shape, ' y_test: ', y_test.shape)


y_train, y_test = y_train['shares'].values, y_test['shares'].values 


# ### Train models on raw data

# Function to train the models

def fit_models(X, y, models):
    out = {}
    for label, model in models.items():
        m = model.fit(X, y)
        p = m.predict(X)
        out[label] = [np.round(score_regression(y, p), 3)]
    return out


# #### List of models to train

models = {
    'OLS': LinearRegression(),     
    'SGDR': SGDRegressor(
        loss='huber',
        penalty='l2',
        max_iter=1000,
        random_state=RANDOM_STATE
    ),    
    'Lasso': Lasso(
        random_state=RANDOM_STATE
    ),    
    'KNN': KNeighborsRegressor(n_jobs=-1),    
    'MLP': MLPRegressor(
        random_state=RANDOM_STATE,
        max_iter = 2000
    ),     
    'RF': RandomForestRegressor(
        n_estimators=100,
        criterion='mse', n_jobs=-1,
        random_state = RANDOM_STATE
    ),
}


# #### Train models on raw data

out = fit_models(X_train, y_train, models)
scores = pd.DataFrame(out, index=['On raw data'])
scores

# #### Define Custom transformers

# Code from the book -Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron-

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names].values       
        

# Define custom log transformer for some unbounded numerical variables

class LogTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log( X + 1)


# #### Define set of columns


# define the list of columns to pre-process
links_features = ['num_hrefs', 'num_self_hrefs',  'self_reference_min_shares',
                  'self_reference_avg_sharess', 'self_reference_max_shares']
# digital media features
dm_features = ['num_imgs', 'num_videos']

cols_log_transform = ['n_tokens_content', 'average_token_length'] + \
     links_features + dm_features + ['num_keywords']

def is_bool_feature(colname):
    if colname.startswith('weekday_is_'):
        return True
    if colname.startswith('data_channel_is_'):
        return True
    if colname.startswith('is_weekend'):
        return True
    return False

# Boolean features
bool_features = [colname for colname in X_train.columns if is_bool_feature(colname)]
# Other numerical features
float_features = [colname for colname in X_train.columns if colname not in bool_features]
# features that will not be Log-transformed and are not boolean
cols_not_to_log_trans_not_bool = [colname for colname in X_train.columns if (colname not in \
    cols_log_transform) and (colname not in bool_features)]

# print(len(cols_log_transform) + len(cols_not_to_log_trans_not_bool) + len(bool_features))
# print(len(bool_features) + len(float_features))
# print(len(X_train.columns))

all_cols = cols_log_transform + cols_not_to_log_trans_not_bool + bool_features


# #### Define pipeline for preprocessing

# Log-transform pipeline
lt_pipe = Pipeline(steps = [
    ('log_selector', FeatureSelector(feature_names=cols_log_transform)),
    ('log_transformer', LogTransformer()),
    ('log_std_scaler', RobustScaler())
])

num_pipe = Pipeline(steps = [
    ('num_selector', FeatureSelector(feature_names=cols_not_to_log_trans_not_bool)),
    ('num_std_scaler', RobustScaler())
])

bool_pipe = Pipeline(steps= [
    ('bool_selector', FeatureSelector(feature_names=bool_features))
])

preprocessor = FeatureUnion(transformer_list = [
    ('lt_pipe', lt_pipe),
    ('num_pipe', num_pipe),
    ('bool_pipe', bool_pipe)
])


# Transform the data and reassign the column names
X_train_pp = preprocessor.fit_transform(X_train)
X_train_pp = pd.DataFrame(X_train_pp, columns=all_cols)

print('X_train:', X_train.shape)
print('X_train_pp:', X_train_pp.shape)

# Train the models on pre-processed data
out = fit_models(X_train_pp, y_train, models)
scores = pd.concat([scores, pd.DataFrame(out, index=['On processed data'])], axis=0)
scores


# ### Cross validation

def fit_models_cv(X, y, models, cv=5):
    n_cv = cv.n_splits if isinstance(cv, KFold) else cv
    rows = ['Training']*n_cv + ['Validation']*n_cv
    out = pd.DataFrame({'Set': rows}, index=rows)
    for label, model in models.items():
        cv_scores = cross_validate(
            estimator=model, X=X, y=y,
            scoring = reg_scorer,
            cv = K_FOLD_SETTINGS,
            return_train_score = True, n_jobs=4
        )
        values = np.append(cv_scores['train_score'], cv_scores['test_score'], axis=0)
        temp = pd.DataFrame({label: np.round(values, 3)}, index=rows)
        out = pd.concat([out, temp], axis=1)
    return pd.DataFrame(out)


# Train models using Cross-Validation
cv_scores = fit_models_cv(X=X_train_pp, y=y_train, models=models, cv=K_FOLD_SETTINGS)

cv_scores.groupby('Set').mean().transpose().plot(
    kind='barh', title='Training and validation scores'
)
plt.savefig(GRAPH_FOLDER +  'scores_train_val.png')

print( cv_scores.groupby('Set').mean() )


# #### Learning curves

# Functions to get the learning curves of all estimators
def learning_curves_models(X, y, models, train_sizes, cv, random_state=0):
    n = len(models)
    cols = 2
    rows = int(np.ceil( n / cols ))
    i = 1
    plt.figure(figsize=(12, 12))
    for label, estimator in models.items():
        lcrv = learning_curve(
            estimator=estimator, X=X, y=y, train_sizes=train_sizes, 
            cv=cv, scoring=reg_scorer, random_state=random_state, n_jobs=-1, 
        )
        ts, train_scores, val_scores = lcrv
        train_scores, val_scores = 1 - train_scores, 1 - val_scores
        
        plt.subplot(rows, cols, i)
        plt.plot(ts, train_scores.mean(1), c='r', label="Training error")
        plt.plot(ts, val_scores.mean(1), c='g', label="Cross validation error")
        plt.title(f'Learning curve {label}')
        if i in [n-1, n]:
            plt.xlabel('Training examples')
        plt.legend(loc="best")
        
        i += 1
    plt.savefig(GRAPH_FOLDER + 'learning_curves.png')


# Learning curves of all the models
learning_curves_models(
    X_train_pp, y_train, models, np.linspace(0.01, 1, 10), 
    K_FOLD_SETTINGS, random_state=RANDOM_STATE
)


# ### Feature selection
#----------------------

# Variable selection using Random Forest feature importance
rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_pp, y_train)

feat_importances = pd.Series(rf.feature_importances_, index = X_train_pp.columns)
feat_importances.nlargest(X_train_pp.shape[1]).sort_values().plot(
    kind='barh', title='Feature importance',figsize=(8,12)
)
plt.savefig(GRAPH_FOLDER + 'feature_selection_rf.png')

# Feature selection using mutual information
mir = mutual_info_regression(
    X=X_train_pp, y=y_train, 
    discrete_features='auto', n_neighbors=5,
    random_state=RANDOM_STATE
)

mutual_info = pd.Series(mir, index=X_train_pp.columns)
mutual_info.sort_values( ascending=True).plot(
    kind='barh', title='Mutual information', figsize=(8,12)
)
plt.savefig(GRAPH_FOLDER + 'feature_selection_mir.png')

# select the common variable among the 1st 40 select by both rf and mir
number_to_select = 45
m = mutual_info.sort_values(ascending=False).index[:number_to_select]
r = feat_importances.sort_values(ascending=False).index[:number_to_select]

selected_features = list(set(m).intersection(set(r)))

X_train_sel = X_train_pp[selected_features]
X_train_sel.shape

print(selected_features)


# ### Model optimization
#-----------------------

# Train Random Forest and KNN
selected_models = {
    'KNN': models['KNN'],    
    'RF': models['RF']
}


# Validation curve function

def validation_curve(X, y, model, param_name, param_range, cv):
    fig, ax = plt.subplots(figsize=(8,4))
    rf_vc = ValidationCurve(
        model, param_name=param_name, scoring=reg_scorer,
        param_range=param_range, cv=cv, n_jobs=-1
    )
    rf_vc.fit(X_train, y_train)
    rf_vc.poof()
    fname = model.__class__.__name__.lower()
    fig.savefig(GRAPH_FOLDER + f"{fname}_vc_{param_name}.png")


# #### Validation curves for RF

# Random Forest validation curve for max_depth
validation_curve(
    X_train_sel, y_train, 
    selected_models['RF'], 'max_depth', 
    np.arange(1,40,3), K_FOLD_SETTINGS
)

# Random Forest validation curve for n_estimators
validation_curve(
    X_train_sel, y_train, selected_models['RF'], 
    'n_estimators', [100, 200, 300, 500], K_FOLD_SETTINGS
)

# Random Forest validation curve for max_features
validation_curve(
    X_train_sel, y_train, selected_models['RF'], 
    'max_features', [1, 2, 3, 5, 7, 11, 13, 17], K_FOLD_SETTINGS
)


# #### Validation curves for KNN

# KNN validation curve for n_neighbors
validation_curve(
    X_train_sel, y_train, selected_models['KNN'], 
    'n_neighbors', [1, 3, 5, 7, 11, 13, 17, 19, 29], K_FOLD_SETTINGS
)

validation_curve(
    X_train_pp, y_train, selected_models['KNN'], 
    'weights', ['uniform', 'distance'], K_FOLD_SETTINGS
)


# #### Grid search for Random Forest

# Optimize  RandomForestRegressor, using the GridSearchCV

rf_grid_params = [
    {'n_estimators': [150, 200, 300], 
     'max_depth': [20, 23, 25, 30],
    },
]

rf_grid_search = GridSearchCV(
    selected_models['RF'], rf_grid_params, 
    cv=K_FOLD_SETTINGS, scoring=reg_scorer, n_jobs=-1,
    return_train_score=True
)
rf_grid_search.fit(X_train_pp[selected_features], y_train)

rf_best_estimator = rf_grid_search.best_estimator_
rf_best_score = rf_grid_search.best_score_
rf_grid_search.best_params_


# #### Grid search for KNN
# Optimize KNN, using the GridSearchCV

knn_grid_params = [
    {'n_neighbors': [3, 5, 7, 9, 11, 13], 
     'metric': ['minkowski', 'mahalanobis', 'seuclidean'],
     'algorithm': ['ball_tree', 'kd_tree']
    }
]

knn_grid_search = GridSearchCV(
    selected_models['KNN'], knn_grid_params, cv=K_FOLD_SETTINGS, 
    scoring=reg_scorer, n_jobs=-1, return_train_score=True
)

knn_grid_search.fit(X_train_pp[selected_features], y_train)

knn_best_estimator = knn_grid_search.best_estimator_
knn_best_score = knn_grid_search.best_score_


# #### The final model
final_model = None
if rf_best_score >= knn_best_score:
    final_model = rf_best_estimator
else:
    final_model = knn_best_estimator


# #### Train the final model on the whole training dataset
print(final_model)
final_model.fit(X_train_pp[selected_features], y_train)


# #### Final predictions

def final_predictions(X_test, y_test, model, preprocessor=None, selected_features=None):
    # Prepare the data
    X_test_pp = preprocessor.transform(X_test) if preprocessor is not None else X_test
    X_test_pp = pd.DataFrame(X_test_pp, columns=X_test.columns)
    # Predictions
    X = X_test_pp if selected_features is None else X_test_pp[selected_features]
    predictions = model.predict(X)
    # Score
    score = score_regression(y_test, predictions)
    return np.round(score * 100, 2)


final_score = final_predictions(
    X_test=X_test, 
    y_test=y_test, 
    model=final_model, 
    preprocessor=preprocessor, 
    selected_features=selected_features
)
print(final_score)


# ## Predictions on X2
X2 = pd.read_csv(DATA_FOLDER + 'X2.csv')

# Pre-process X2
X2_pp = preprocessor.transform(X2)
X2_pp = pd.DataFrame(X2_pp, columns=all_cols)
Y2 = final_model.predict(X2_pp[selected_features])

# Save the predictions
pd.DataFrame(Y2).to_csv('Y2.csv')


t1 = time.time()

print('Total running time:', t1-t0)
print((t1-t0)/60, 'minutes')

