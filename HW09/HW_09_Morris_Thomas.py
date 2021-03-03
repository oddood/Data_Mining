"""
Author: Thomas Morris | tcm1998@rit.edu
Date: 17 April 2020
"""

import pandas
import os.path
import sys
from random import randint
import docx
from matplotlib import pyplot
import matplotlib.lines as lines
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def read_data_file(data_file_name):
    """Given a file name, locate, open, and read the file. The file is assumed
    to be a CSV file and its data will be parsed accordingly and returned as a
    pandas dataframe.

    :param data_file_name: <str> name of CSV file to read data from
    :return: <Pandas.DataFrame> all data read from data_file_name
    """
    # Assume the data file is in the current working directory.
    cwd = os.getcwd()
    data_file_path = os.path.join(os.getcwd(), data_file_name)

    # If it isn't present, check the parent folder.
    if not os.path.exists(data_file_path):
        super_folder = os.path.dirname(cwd)
        data_file_path = os.path.join(super_folder, data_file_name)

        # If it still can't be found, exit.
        if not os.path.exists(data_file_path):
            sys.exit("Provided data file could not be found.")

    # We have found the file and can extract the data.
    data = pandas.read_csv(data_file_path, delimiter=',')

    # Cast everything to integer.
    return data


def balance_data(data):
    """Count the occurrences of the two classes: Assam and Bhuttan, and, if the
    two are unbalanced, randomly remove rows from the larger group until its
    size matches the smaller.

    :param data: <Pandas.DataFrame> Data of Assam and Bhutan snowmen to be
        balanced randomly
    :return: <Pandas.DataFrame> Data with equal frequency of the Assam and
        Bhutan classes
    """
    # Group the data by classification and record the classes' frequencies.
    class_sizes = data.groupby(['Class']).size()
    a_count = class_sizes['Assam']
    b_count = class_sizes['Bhuttan']

    imbalance = a_count - b_count
    if imbalance > 0:
        # If the difference is positive, class Assam is too large.
        majority = data[data.Class == 'Assam']
        minority = data[data.Class == 'Bhuttan']
    elif imbalance < 0:
        # If the difference is negative, class Bhuttan is too large.
        majority = data[data.Class == 'Bhuttan']
        minority = data[data.Class == 'Assam']
    else:
        # The data is already balanced.
        return data

    # Ensure the imbalance is a positive number.
    imbalance = abs(imbalance)

    # Reset the row index for the group we intend to remove rows from randomly.
    majority = majority.reset_index(drop=True)
    dropped_indexes = set()

    # Remove rows until the imbalance is accounted for.
    while imbalance > 0:
        # Generate a random row number.
        random_index = randint(0, majority.shape[0] - 1)

        # If we already removed this index, continue.
        if random_index in dropped_indexes:
            continue

        # Drop the row with this index value and decrement the imbalance.
        try:
            majority.drop(index=random_index)
            imbalance -= 1
        except Exception as e:
            print(e)
        finally:
            dropped_indexes.add(random_index)

    # Recombine the majority and minority and reset the row indexing.
    return pandas.concat([majority, minority], ignore_index=True)


def feature_generation(data):
    """Add additional features to the data which are combinations of various
    features, such as Tail Length minus Hair Length.

    :param data: <Pandas.DataFrame> data to augment
    :return: The original data with additional columns.
    """
    col_idx = data.shape[1] - 1

    # Tail length minus hair length.
    data.insert(loc=col_idx, column='TailLessHair',
                value=data['TailLn'] - data['HairLn'])
    col_idx += 1

    # Tail length minus bang length.
    data.insert(loc=col_idx, column='TailLessBang',
                value=data['TailLn'] - data['BangLn'])
    col_idx += 1

    # How much longer is the body hair compared to the bangs.
    data.insert(loc=col_idx, column='ShagFactor',
                value=data['HairLn'] - data['BangLn'])
    col_idx += 1

    # Tail length plus hair length.
    data.insert(loc=col_idx, column='TailAndHair',
                value=data['TailLn'] + data['HairLn'])
    col_idx += 1

    # Tail length plus bang length.
    data.insert(loc=col_idx, column='TailAndBangs',
                value=data['TailLn'] + data['BangLn'])
    col_idx += 1

    # Hair length plus bang length.
    data.insert(loc=col_idx, column='HairAndBangs',
                value=data['HairLn'] + data['BangLn'])
    col_idx += 1

    # Hair len. + bang len. + tail len.
    data.insert(loc=col_idx, column='AllLengths',
                value=data['HairLn'] + data['BangLn'] + data['TailLn'])
    col_idx += 1

    # Rock climbing factor called Ape Factor for how much above your height you
    # can reach. It is arm reach minus height.
    data.insert(loc=col_idx, column='ApeFactor',
                value=data['Reach'] - data['Ht'])
    col_idx += 1

    # Random combination: height minus age.
    data.insert(loc=col_idx, column='Height-Age',
                value=data['Ht'] - data['Age'])
    # col_idx += 1

    return data


def print_cross_correlation(data):
    """Print a cross-correlation matrix into a Word .docx file.

    :param data: <Pandas.DataFrame>
    """
    # Make printing a table this big a bit cleaner.
    pandas.set_option('display.max_rows', 25)
    pandas.set_option('display.max_columns', 25)
    pandas.set_option('display.width', 110)

    # i am not sure how you are getting your data, but you said it is a
    # pandas data frame
    print_data = data.replace({'Class': {'Assam': -1, 'Bhuttan': 1}})
    df = print_data.corr()
    df = df.round(decimals=3)

    print(df)

    # open an existing document
    doc = docx.Document()

    # add a table to the end and create a reference variable
    # extra row is so we can add the header row
    t = doc.add_table(df.shape[0] + 1, df.shape[1])

    # add the header rows.
    for j in range(df.shape[-1]):
        t.cell(0, j).text = df.columns[j]

    # add the rest of the data frame
    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            t.cell(i + 1, j).text = str(df.values[i, j])

    # save the doc
    # doc.save('Corr.docx')
    return df


def best_features_cc(cross_corr):
    """Find the best positive and negative correlations with the target
    attribute: Class, of the given cross-correlation matrix, then return the
    attributes' names.

    :param cross_corr: <Pandas.DataFrame> Pandas cross-correlation.
    :return: <Tuple(str, str)> Names of the attributes most positively and
        negatively correlated with attributes Class, respectively.
    """
    # Extract the target series.
    target = cross_corr['Class']

    # Drop its correlation to itself.
    target = target.drop(labels='Class')

    # Best attributes' names.
    positive = None
    negative = None

    # Best positive correlation and best negative correlation.
    best_positive_corr = 0.0
    best_negative_corr = 0.0

    for attribute in target.index:
        # Check if its better than our best positive correlation.
        if target[attribute] > best_positive_corr:
            best_positive_corr = target[attribute]
            positive = attribute

        # Check if its better than our best negative correlation.
        if target[attribute] < best_negative_corr:
            best_negative_corr = target[attribute]
            negative = attribute

    # Print the output.
    print("{0} is the most POSITIVELY correlated attribute with {1}"
          .format(positive, best_positive_corr))
    print("{0} is the most NEGATIVELY correlated attribute with {1}"
          .format(negative, best_negative_corr))

    # Return the two attributes in order of greatest correlation.
    if best_positive_corr > abs(best_negative_corr):
        return positive, negative
    else:
        return negative, positive


def plot_features(data, feature1, feature2):
    # Collect the x, y data for the given features for Assam and Bhuttan data
    # separately.
    assam_xdata = data[data.Class == 'Assam'][feature1]
    assam_ydata = data[data.Class == 'Assam'][feature2]
    bhuttan_xdata = data[data.Class == 'Bhuttan'][feature1]
    bhuttan_ydata = data[data.Class == 'Bhuttan'][feature2]

    # Calculate the centers of the two distributions.
    assam_mean = (assam_xdata.mean(), assam_ydata.mean())
    bhuttan_mean = (bhuttan_xdata.mean(), bhuttan_ydata.mean())

    # Plot the Assam and Bhuttan data as empty circles.
    fig, axes = pyplot.subplots()
    axes.scatter(assam_xdata, assam_ydata, s=3**2,
                 facecolors='none', edgecolors='r', linewidths=0.5)
    axes.scatter(bhuttan_xdata, bhuttan_ydata, s=3**2,
                 facecolors='none', edgecolors='b', linewidths=0.5)
    axes.set_xlabel(feature1)
    axes.set_ylabel(feature2)

    # Add a line segment connecting the two centers.
    # This is the projection vector for the two distributions.
    axes.add_line(lines.Line2D(xdata=[assam_mean[0], bhuttan_mean[0]],
                               ydata=[assam_mean[1], bhuttan_mean[1]],
                               linewidth=2.0, color='g'))

    # What follows is a tedious process of plotting a decision boundary which
    # is the perpendicular bisector of the projection vector.

    # Midpoint of centers
    midpoint_x = (assam_mean[0] + bhuttan_mean[0])/2
    midpoint_y = (assam_mean[1] + bhuttan_mean[1])/2

    # Slope of the decision boundary
    delta_y = assam_mean[1] - bhuttan_mean[1]
    delta_x = assam_mean[0] - bhuttan_mean[1]
    db_slope = -(delta_x / delta_y)
    print("Projection Vector slope ~= {0}".format(delta_y/delta_x))
    print("Decision Boundary slope ~= {0}".format(db_slope))

    # Decision boundary y-intercept value
    db_intercept = midpoint_y - (db_slope * midpoint_x)

    # Get the maximum and minimum values for feature 2.
    f2_max = data[feature2].max()
    f2_min = data[feature2].min()

    # Use the range of feature 2 to determine the endpoints of our decision
    # boundary.
    db_yvals = [f2_max, f2_min]
    db_xvals = [(f2_max - db_intercept)/db_slope,
                (f2_min - db_intercept)/db_slope]

    # Plot the decision boundary.
    axes.add_line(lines.Line2D(xdata=db_xvals, ydata=db_yvals,
                               linewidth=2.0, color='y'))

    # Set a legend and display the figure.
    axes.legend(['Projection Vector', 'Decision Boundary', 'Assam', 'Bhuttan'])
    pyplot.show()

    return db_slope, db_intercept


def test_decision_boundary(data, feature1, feature2, slope, intercept):
    # Correctly guessed Assam, incorrectly guessed Assam, etc.
    correct_assam = 0
    incorrect_assam = 0
    correct_bhuttan = 0
    incorrect_bhuttan = 0

    for idx, series in data.iterrows():
        # Using the row's y value, calculate x with the given slope and
        # intercept values for the decision boundary.
        boundary_x = (series[feature2] - intercept)/slope

        guess = ''

        # Use the decision boundary to make a guess.
        if series[feature1] < boundary_x:
            # Left of the boundary -> Assam
            guess = 'Assam'
        elif series[feature1] > boundary_x:
            # Right of the boundary -> Bhuttan
            guess = 'Bhuttan'
        else:
            # If it is on the boundary, place it in the more common class.
            if correct_assam + incorrect_assam > \
                    correct_bhuttan + incorrect_bhuttan:
                guess = 'Assam'
            else:
                guess = 'Bhuttan'

        # Figure out which part of the confusion matrix this round goes to.
        if guess == 'Assam' and series['Class'] == 'Assam':
            correct_assam += 1
        elif guess == 'Assam' and series['Class'] != 'Assam':
            incorrect_assam += 1
        elif guess == 'Bhuttan' and series['Class'] == 'Bhuttan':
            correct_bhuttan += 1
        else:
            incorrect_bhuttan += 1

    print("\t\t\t\t\t\tGuess")
    print("\t\t\t\t\tAssam\tBhuttan")
    print("\tAssam\t\t\t| {0} |\t| {1} |".format(correct_assam, incorrect_bhuttan))
    print("Actual Class")
    print("\tBhuttan\t\t\t| {0} |\t| {1} |".format(incorrect_assam, correct_bhuttan))


def brute_force_lda(data):
    """Build a linear discriminant analysis classifier for every possible pair
    of features. Return the best classifier and its features and print out
    results for both the first and second best classifiers.

    :param data: <Pandas.DataFrame>
    :return: <sklearn.LinearDiscriminantAnalysis, List<str>> Classifier and a
        list of the features it uses.
    """
    num_rows = data.shape[0]
    num_features = data.shape[1] - 1

    data = data.replace({'Class': {'Assam': -1, 'Bhuttan': 1}})
    target_data = data['Class']
    training_data = data.drop(['Class'], axis=1)

    col_names = training_data.columns

    first_pair = None
    first_place = 0.0
    second_pair = None
    second_place = 0.0

    best_lda = None

    for idx1 in range(len(col_names)-1):
        for idx2 in range(idx1 + 1, len(col_names)):
            # Make a new LDA object.
            classifier = LinearDiscriminantAnalysis()

            # Train it with only two of the available features.
            classifier.fit(training_data[[col_names[idx1], col_names[idx2]]],
                           target_data)

            # Test the classifier with the same training data.
            predictions = classifier.predict(training_data[[col_names[idx1], col_names[idx2]]])

            correct = 0

            # Record how many times it was correct.
            for idx, series in data.iterrows():
                if series['Class'] == predictions[idx]:
                    correct += 1

            # Update our records accordingly.
            accuracy = correct/num_rows

            if accuracy > first_place:
                # Move first down to second.
                second_place = first_place
                second_pair = first_pair

                # Then replace first.
                first_place = accuracy
                first_pair = [col_names[idx1], col_names[idx2]]
                best_lda = classifier

            elif accuracy > second_place:
                # It was only better than second, so only replace second.
                second_place = accuracy
                second_pair = [col_names[idx1], col_names[idx2]]

    print("Best pair is {0} and {1} with {2}".format(first_pair[0],
                                                     first_pair[1],
                                                     first_place))
    print("Second place is {0} and {1} with {2}".format(second_pair[0],
                                                        second_pair[1],
                                                        second_place))
    return best_lda, first_pair


def main():
    classified_data = read_data_file(sys.argv[1])
    balanced_data = balance_data(classified_data)
    enhanced_data = feature_generation(balanced_data)
    cc_matrix = print_cross_correlation(enhanced_data)
    (feature1, feature2) = best_features_cc(cc_matrix)
    (slope, intercept) = plot_features(enhanced_data, feature1, feature2)
    test_decision_boundary(enhanced_data, feature1, feature2,
                           slope, intercept)
    (classifier, lda_features) = brute_force_lda(enhanced_data)

    # Use the LDA classifier on the unclassified data and save the results to
    # a file.
    unclassified_data = read_data_file(sys.argv[2])
    enhanced_unclassified = feature_generation(unclassified_data)
    predictions = classifier.predict(
        enhanced_unclassified[[lda_features[0], lda_features[1]]])

    with open('HW_09_Classified_Results.csv', mode='w') as f:
        for p in predictions:
            if p < 0:
                f.write('-1\n')
            else:
                f.write('+1\n')
        f.flush()


if __name__ == "__main__":
    main()
