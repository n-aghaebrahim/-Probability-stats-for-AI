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
