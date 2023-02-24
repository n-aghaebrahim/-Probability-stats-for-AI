param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}



def grid_search_rfc(X_train, y_train, param_grid, cv=5):
    """
    Performs a grid search on a random forest classifier using scikit-learn's GridSearchCV.

    Parameters:
        X_train (np.array): The training set features.
        y_train (np.array): The training set target values.
        param_grid (dict): The grid of hyperparameters to search over.
        cv (int, optional): The number of cross-validation folds. Default is 5.

    Returns:
        scores (list): A list of mean cross-validation scores for each hyperparameter combination.
    """
    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(rfc, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)

    scores = grid_search.cv_results_['mean_test_score']

    return scores
