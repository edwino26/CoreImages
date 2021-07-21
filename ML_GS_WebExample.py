# %%  
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o-o-o-o-o-o-o-  GRID SEARCH o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
#-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-


rgr.get_params()  #Parameters used in fitted estimator

# A search consists of:
# an estimator (regressor or classifier such as sklearn.svm.SVC());
# a parameter space;
# a method for searching or sampling candidates;
# a cross-validation scheme; and
# a score function.


print("Available hyperparameters for this estimator: ", rgr.get_params()  )
tuned_parameters = [{'alpha':[0.05, 0.5, 0.9], 'n_estimators': [1, 10, 100, 300], 'max_depth': [3, 9, 27], 'learning_rate': [0.1, 1, 3]},
 ]
 
#Scoring:  Methods used to determine the best model depending on problem type
#Ref: https://scikit-learn.org/stable/modules/model_evaluation.html 
# Why RMSE is better to avoid large errors in higher values (bright parts of UV photos)
# https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
print("--------------------------------")
print("Available Scorers for GridSearch: ", sorted(SCORERS.keys()))
scores = ['neg_root_mean_squared_error']


for score in scores:
    print()
    print("# Tuning hyper-parameters for %s" % score)
    print()

    gsr = GridSearchCV(
        GradientBoostingRegressor(), tuned_parameters, scoring=score  #use cv=None for default 5-fold cross validation
    )
    gsr.fit(X, np.ravel(y))

    print("Best parameters set found on development set:")
    print()
    print(gsr.best_params_)
    print("Best RMSE: ", gsr.best_score_)
    print()
    print("Grid scores on development set:")
    print()
    means = gsr.cv_results_['mean_test_score']
    stds = gsr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gsr.cv_results_['params']):
        if abs(mean) < 1000:  #Remove "exploding errors"
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("Done: the model is trained on the full development set.")
    print("Done: the scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, rgr.predict(X_test)