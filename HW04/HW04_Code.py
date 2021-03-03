"""
Author: Thomas Morris | tcm1998
Date: 8 February 2020
"""

import pandas
import os.path
import sys
from datetime import date

TRAINED_FILE_NAME = 'HW_03_Morris_Thomas_Trained.py'
TARGET_ATTR = 'Sickness'


def build_classifier(file_name):
    # Assume the data file is in the current working directory.
    cwd = os.getcwd()
    data_file_path = os.path.join(os.getcwd(), file_name)

    # If it isn't present, check the parent folder.
    if not os.path.exists(data_file_path):
        super_folder = os.path.dirname(cwd)
        data_file_path = os.path.join(super_folder, file_name)

        # If it still can't be found, exit.
        if not os.path.exists(data_file_path):
            sys.exit("Data file could not be found.")

    # Read the CSV file into a Pandas dataframe.
    training_data = pandas.read_csv(data_file_path, delimiter=',', index_col=0)
    # ccs = training_data[training_data['FavCookie'] == 'Chocolate chip']
    # print(ccs['BreakFastDrink'].value_counts())

    macs = training_data[training_data['FavCookie'] == 'Macadamia nut']
    print(macs['MostFavTopping'].value_counts())
    print(macs['LeastFavTopping'].value_counts())

    '''
    ex = training_data.drop(columns=['log2(1024)', 'Six fingers?', 'First Compiler', 'SchoolWoman', 'Diapers?',
                                     'Hearing issues?'])

    # Delete the GuestID column, as it serves no purpose here.
    # del training_data['GuestID']

    # Calculate the cross-correlational coefficients.
    
    ex = ex.replace({'Snack Food': {'Popcorn': 1, 'M&Ms or candy': 0,
                                               'Tortilla chips': 2, 'Carrots and cellery': 3},
                                'Frequency of Exercise': {'0 Exercise? Never touch the stuff.': '0'},
                                'Elevator Use?': {'I usually ride the elevator.': 1,
                                                  'Wait. There is an elevator in the building?': 1,
                                                  ' the stairs are always faster.\"': 3,
                                                  'I always ride the elevator.': 0}})

    ex = ex.astype({'Frequency of Exercise': int})

    print(ex[['Elevator Use?', 'Frequency of Exercise']].corr())
    # print(ex[['Snack Food', 'Frequency of Exercise']].describe())
    # cross_correlation_matrix = ex[['Frequency of Exercise', 'Elevator Use?']].astype({'Frequency of Exercise': int}).corr()

    # Print the cross-correlational matrix.
    # print(cross_correlation_matrix)
    '''

    '''
    ex.astype({'ELAPSED_TIME': int})
    fast_testers = ex[training_data['ELAPSED_TIME'] <= 20]
    slow_testers = ex[training_data['ELAPSED_TIME'] > 20]

    for column_name in fast_testers.columns:
        fast_frequencies = fast_testers[column_name].value_counts(normalize=True)
        slow_frequencies = slow_testers[column_name].value_counts(normalize=True)

        fast_value = fast_frequencies.iloc[0]
        slow_value = slow_frequencies.iloc[0]

        if fast_value >= 0.5 and slow_value < 0.5:
            print('{0}: fast -> {1}'.format(column_name, fast_frequencies.index[0],
                                                              slow_frequencies.index[0]))
        elif fast_value < 0.5 and slow_value >= 0.5:
            print('{0}: slow -> {2}'.format(column_name, fast_frequencies.index[0],
                                                              slow_frequencies.index[0]))
        elif fast_value >= 0.5 and slow_value >= 0.5 and fast_frequencies.index[0] != slow_frequencies.index[0]:
            print('{0}: fast -> {1}; slow -> {2}'.format(column_name, fast_frequencies.index[0],
                                                              slow_frequencies.index[0]))

    print(fast_testers.corr())
    print(slow_testers.corr())
    '''


def find_one_rule(dataframe):
    # All attributes except sickness.
    col_index = dataframe.columns.drop(TARGET_ATTR)

    best_attr = ''
    best_accuracy = 0.0
    inverse = False

    row_count = dataframe.shape[0]

    for food in col_index:
        # Use an aggregate function to count the rows in which this column
        # was equal to the Sickness column.
        count = dataframe[dataframe[food] == dataframe[TARGET_ATTR]].count()

        correct = count[food]

        # If this attribute has better accuracy, save it.
        if correct/row_count > best_accuracy:
            best_attr = food
            best_accuracy = correct/row_count
            inverse = False
        # Also save it if it has the best inaccuracy.
        elif (1 - correct/row_count) > best_accuracy:
            best_accuracy = 1 - correct/row_count
            best_attr = food
            inverse = True

    # print(best_attr, best_accuracy)
    return best_attr, inverse


def main(training_file_name):
    """Main function.
    Determine the best classification attribute and threshold, then write the
    training program using this information.

    :param training_file_name: <str> name of the training data file
    :return: None
    """
    build_classifier(training_file_name)


if __name__ == "__main__":
    main('CS420_and_720_Obtuse_data_Anonymous_v0322.csv')
