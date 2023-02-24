def preprocess_df(df, target_col, test_size=0.2, random_state=0):
    """
    Preprocesses a pandas dataframe by standard scaling the features and splitting
    the data into training and test sets using cross validation.

    Parameters:
        df (pd.DataFrame): The input dataframe to be preprocessed.
        target_col (str): The name of the target column in the dataframe.
        test_size (float, optional): The size of the test set as a fraction of the data. Default is 0.2.
        random_state (int, optional): The seed for the random number generator used in train_test_split. Default is 0.

    Returns:
        X_train (np.array): The preprocessed training set features.
        X_test (np.array): The preprocessed test set features.
        y_train (np.array): The training set target values.
        y_test (np.array): The test set target values.
    """
    # Separate the target column from the features
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Save the training and test sets as separate CSV files
    pd.DataFrame(X_train, columns=df.drop(columns=[target_col]).columns).to_csv("train_features.csv", index=False)
    pd.DataFrame(X_test, columns=df.drop(columns=[target_col]).columns).to_csv("test_features.csv", index=False)
    pd.DataFrame(y_train, columns=[target_col]).to_csv("train_target.csv", index=False)
    pd.DataFrame(y_test, columns=[target_col]).to_csv("test_target.csv", index=False)

    return X_train, X_test, y_train, y_test





