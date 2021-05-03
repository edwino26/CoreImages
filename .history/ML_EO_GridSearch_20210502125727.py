
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
    tuned_parameters = [{'C':np.linspace(0.01,3, 10), 'epsilon':np.linspace(0.01,0.3, 4), 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_
    
    
def  RandomForestRegressorGS(X, y):
    #  'bootstrap': True,
    #  'ccp_alpha': 0.0,
    #  'criterion': 'mse',
    #  'max_depth': None,
    #  'max_features': 'auto',
    #  'max_leaf_nodes': None,
    #  'max_samples': None,
    #  'min_impurity_decrease': 0.0,
    #  'min_impurity_split': None,
    #  'min_samples_leaf': 1,
    #  'min_samples_split': 2,
    #  'min_weight_fraction_leaf': 0.0,
    #  'n_estimators': 100,
    #  'n_jobs': None,
    #  'oob_score': False,
    #  'random_state': None,
    #  'verbose': 0,
    #  'warm_start': False
    rgr = RandomForestRegressor()
    rgr.get_params() 
    tuned_parameters = [{'n_estimators':np.array([10, 50, 100, 500, 1000])}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_

 
def  GradientBoostingRegressorGS(X, y):
    # 'alpha': 0.9,
    #  'ccp_alpha': 0.0,
    #  'criterion': 'friedman_mse',
    #  'init': None,
    #  'learning_rate': 0.1,
    #  'loss': 'ls',
    #  'max_depth': 3,
    #  'max_features': None,
    #  'max_leaf_nodes': None,
    #  'min_impurity_decrease': 0.0,
    #  'min_impurity_split': None,
    #  'min_samples_leaf': 1,
    #  'min_samples_split': 2,
    #  'min_weight_fraction_leaf': 0.0,
    #  'n_estimators': 100,
    #  'n_iter_no_change': None,
    #  'random_state': None,
    #  'subsample': 1.0,
    #  'tol': 0.0001,
    #  'validation_fraction': 0.1,
    #  'verbose': 0,
    #  'warm_start': False
    rgr = GradientBoostingRegressor()
    rgr.get_params() 
    tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10), 'learning_rate':np.linspace(0.01,3, 10), 'n_estimators':np.array([10, 50, 100, 500, 1000])}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_    


def  MLPRegressorGS(X, y):
    architecture = [x for x in itertools.product((10,50,100,200),repeat=4)]
    #  'activation': 'relu',
    #  'alpha': 0.0001,
    #  'batch_size': 'auto',
    #  'beta_1': 0.9,
    #  'beta_2': 0.999,
    #  'early_stopping': False,
    #  'epsilon': 1e-08,
    #  'hidden_layer_sizes': (100,),
    #  'learning_rate': 'constant',
    #  'learning_rate_init': 0.001,
    #  'max_fun': 15000,
    #  'max_iter': 200,
    #  'momentum': 0.9,
    #  'n_iter_no_change': 10,
    #  'nesterovs_momentum': True,
    #  'power_t': 0.5,
    #  'random_state': None,
    #  'shuffle': True,
    #  'solver': 'adam',
    #  'tol': 0.0001,
    #  'validation_fraction': 0.1,
    #  'verbose': False,
    #  'warm_start': False
    rgr = MLPRegressor()
    rgr.get_params() 
    tuned_parameters = [{'hidden_layer_sizes':(30,50,100), 'learning_rate':np.linspace(0.00001,0.0001, 0.001), 'alpha':np.linspace(0.01,0.99, 10) }]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    print(gsr.best_params_)
    return gsr.best_estimator_    





# %%
