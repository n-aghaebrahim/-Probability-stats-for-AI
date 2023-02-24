def perform_eda(df):
    """
    A function that performs Exploratory Data Analysis on a given pandas DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to perform EDA on.
    
    Returns:
    None
    """
    # print the first 5 rows of the data
    print("First 5 rows:")
    print(df.head())
    print("\n")

    # print the shape of the data
    print("Data Shape:", df.shape)
    print("\n")

    # print the data types of each column
    print("Data Types:")
    print(df.dtypes)
    print("\n")

    # print the summary statistics of the numerical columns
    print("Summary Statistics:")
    print(df.describe())
    print("\n")

    # print the number of missing values in each column
    print("Missing Values:")
    print(df.isnull().sum())
    print("\n")




def explore_data(df):
    """
    Create plots to explore the given DataFrame.
    """
    
    # Plot histogram of all numerical columns
    num_cols = df.select_dtypes(include=['float', 'int']).columns
    num_cols_count = len(num_cols)
    fig, axs = plt.subplots(num_cols_count, 1, figsize=(8, 4*num_cols_count))
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axs[i])
        axs[i].set_xlabel(col)
    
    plt.show()
    
    # Plot correlation matrix
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    plt.show()
    
    # Plot pairplot of all numerical columns
    sns.set(font_scale=0.8)
    sns.pairplot(df.select_dtypes(include=['float', 'int']))
    plt.show()

    # Plot distribution plot of all numerical columns
    num_cols = df.select_dtypes(include=['float', 'int']).columns
    num_cols_count = len(num_cols)
    fig, axs = plt.subplots(num_cols_count, 1, figsize=(8, 4*num_cols_count))
    for i, col in enumerate(num_cols):
        sns.displot(data=df, x=col, kde=True, ax=axs[i])
        axs[i].set_xlabel(col)
    
    plt.show()
    
    # Plot boxplot of each numerical column
    num_cols = df.select_dtypes(include=['float', 'int']).columns
    num_cols_count = len(num_cols)
    fig, axs = plt.subplots(num_cols_count, 1, figsize=(8, 4*num_cols_count))
    for i, col in enumerate(num_cols):
        df.boxplot(column=col, ax=axs[i])
    
    plt.show()
    
    # Plot bar chart of each categorical column
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols_count = len(cat_cols)
    fig, axs = plt.subplots(cat_cols_count, 1, figsize=(8, 4*cat_cols_count))
    for i, col in enumerate(cat_cols):
        df[col].value_counts().plot(kind='bar', ax=axs[i])
    
    plt.show()



def t_test_p_value(dataframe, target_column):
    """
    Perform a t-test and calculate the p-value for each feature against the target column in a pandas dataframe.
    
    Parameters:
    dataframe (pd.DataFrame): The dataframe to perform the t-test on.
    target_column (str): The name of the target column to use in the t-test.
    
    Returns:
    dict: A dictionary of t-statistics and p-values for each feature against the target column.
    """
    
    # Create an empty dictionary to store the results
    results = {}
    
    # Iterate over each column in the dataframe
    for column in dataframe.columns:
        if column != target_column:
            # Perform a t-test between the current column and the target column
            t_stat, p_value = stats.ttest_ind(dataframe[column], dataframe[target_column], equal_var=False)
            
            # Store the t-statistic and p-value in the results dictionary
            results[column] = {"t-statistic": round(t_stat,3), "p-value": round(p_value,3)}
    
    # Return the results dictionary
    return results



