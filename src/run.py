import os
from data import data
from data import analys_data

# .csv files
red_wine = 'train_data/winequality/winequality-red.csv'
white_wine = 'train_data/winequality/winequality-white.csv'

def main():

    # Create directories
    if not os.path.exists('logs/plots'):
        os.makedirs('logs/plots')

    if not os.path.exists('logs/analysis'):
        os.makedirs('logs/analysis')

    # Generate dataframe
    wine_df = data.combine_data(red_wine_data_path=red_wine,
                                white_wine_data_path=white_wine
                                )


    # preprocess the dataframe
    X_train, X_test, y_train, y_test = data.preprocess_df(wine_df)
    with open ('logs/analysis/eda.csv', 'w') as f:
        f.write(f'\n\nx train shape:  {X_train.shape}\n')
        f.write(f'X test shape: {X_test.shape}\n')
        f.write(f'y train shape: {y_train.shape}\n')
        f.write(f'y test shape: {y_test.shape}\n')



    # analysing data
    analys_data.perform_eda(wine_df)
    analys_data.explore_data(wine_df)
    analys_data.t_test_p_value(dataframe=wine_df,
                                target_column='quality')
   

main()
