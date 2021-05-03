
# %%
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o--o-  GRID SEARCH for ALL ALGORITHMS o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
score_RMSE = 'neg_root_mean_squared_error'
# Lasso

def lassoGS(X, y):
    # {'alpha': 1.0,
    #  'copy_X': True,
    #  'fit_intercept': True,
    #  'max_iter': 1000,
    #  'normalize': False,
    #  'positive': False,
    #  'precompute': False,
    #  'random_state': None,
    #  'selection': 'cyclic',
    #  'tol': 0.0001,
    #  'warm_start': False}
    rgr = linear_model.Lasso()
    rgr.get_params() 
    tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10)}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_

def ElasticNetGS(X, y):
    #  {'alpha': 1.0,
    #  'copy_X': True,
    #  'fit_intercept': True,
    #  'l1_ratio': 0.5,
    #  'max_iter': 1000,
    #  'normalize': False,
    #  'positive': False,
    #  'precompute': False,
    #  'random_state': None,
    #  'selection': 'cyclic',
    #  'tol': 0.0001,
    #  'warm_start': False}
    rgr = linear_model.ElasticNet()
    rgr.get_params() 
    tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10), 'l1_ratio':np.linspace(0.01,0.99, 10)}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_


def  RidgeGS(X, y):
    #  {'alpha': 1.0,
    #  'copy_X': True,
    #  'fit_intercept': True,
    #  'max_iter': None,
    #  'normalize': False,
    #  'random_state': None,
    #  'solver': 'auto',
    #  'tol': 0.001}
    rgr = linear_model.Ridge()
    rgr.get_params() 
    tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10)}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_


def  SVRGS(X, y):
    #  {'C': 1.0,
    #  'cache_size': 200,
    #  'coef0': 0.0,
    #  'degree': 3,
    #  'epsilon': 0.1,
    #  'gamma': 'scale',
    #  'kernel': 'rbf',
    #  'max_iter': -1,
    #  'shrinking': True,
    #  'tol': 0.001,
    #  'verbose': False}
    rgr = SVR()
    rgr.get_params() 
    tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10)}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_


# %%
