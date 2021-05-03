
# %%
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o--o-  GRID SEARCH for ALL ALGORITHMS o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
score_RMSE = 'neg_root_mean_squared_error'
# Lasso

def lassoGS(X, y):
    rgr = linear_model.Lasso()
    rgr.get_params() 
    tuned_parameters = [{'alpha':np.linspace(0.01,0.99, 10)}]
    gsr= GridSearchCV(rgr, tuned_parameters, scoring=score_RMSE, refit=True)
    gsr.fit(X, y)
    return gsr.best_params_





# %%
