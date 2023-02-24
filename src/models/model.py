def best_model(X_train, X_test, y_train, y_test, target_col):
    """
    X_train:
    X_test:
    y_train: 
    y_test:
    target_col: Target column
    """

    # Initialize a list of classifiers
    classifiers = [LogisticRegression(), 
                   KNeighborsClassifier(), 
                   DecisionTreeClassifier(), 
                   RandomForestClassifier(), 
                   SVC(), 
                   GaussianNB()]
    
    # Initialize a list to store the accuracy scores
    classifier = []
    scores = []
    pred = []

    
    # Loop over each classifier and fit the model to the training data
    for clf in classifiers:
        model_name = type(clf).__name__
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = round(accuracy_score(y_test, y_pred),3)
        scores.append(score)
        classifier.append(model_name)
        pred.append(y_pred)

    # Output Classifier table
    model_scores = pd.DataFrame(
         {'Model': classifier,
          'Score': scores   
         }) 
    model_scores = model_scores.sort_values(by='Score', ascending=False).reset_index()


    
    # Find the index of the classifier with the highest accuracy score
    best_index = np.argmax(scores)
    
    # Return the classifier with the highest accuracy score
    return print('Best Classifier is: ', classifiers[best_index],'\n\n', model_scores), pred



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
