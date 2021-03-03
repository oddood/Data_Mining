"""
Author: Thomas Morris
Date: 1 February 2020
Purpose:
    HW01 on clustering. Read data sets, perform exploratory analysis, and
    divide a dataset into two clusters using Otsu's method for 1D clustering
    and, separately, a cost function method derived from Otsu's method.
    Finally, plot the data points against their variance and circle the
    threshold point.
"""

import numpy
import os.path
from matplotlib import pyplot

mystery_data_file = 'Mystery_Data_2195.csv'
abominable_data_file = 'Abominable_Data_For_Clustering__v44.csv'
age_bin_size = 2
norm_factor = 100


def read_mystery_data():
    """Read the mystery data file and return its data as a numpy.array.

    :return: numpy.ndarray of the data cast to integers
    """
    mystery_data = []
    with open(os.path.join('..', mystery_data_file), mode='r') as data_file:
        for line in data_file:
            # Remove any whitespace from each line and save the data as
            # integers.
            value = line.strip(" \n")
            if value.isnumeric():
                mystery_data.append(int(value))
    return mystery_data


def read_abominable_data():
    """Read the Abominable snowfolk data file and return its age data as a
    numpy.ndarray.

    :return: numpy.ndarray of the floating point age data
    """
    snowfolk_ages = []
    with open(os.path.join('..', abominable_data_file), mode='r') as data_file:
        for line in data_file:
            # Remove any whitespace from each line.
            clean_line = line.strip(" \n")
            age_and_height = clean_line.split(',')
            if not age_and_height[0].isalpha():
                snowfolk_ages.append(float(age_and_height[0]))
    data_array = numpy.array(snowfolk_ages)
    data_array.sort()
    return data_array


def mystery_exploratory_analysis(mystery_data):
    """Calculate the mean and standard deviation of the 1D array, then
    recalculate them after removing the last value.

    :param mystery_data: numpy.ndarray of numbers
    :return: None
    """
    # Read the data and compute the mean and standard deviation.
    numpy_data_original = numpy.array(mystery_data)
    true_mean = numpy_data_original.mean()
    true_std = numpy_data_original.std()
    print("First Mean = {0}\nFirst Standard Deviation = {1}"
          .format(true_mean, true_std))

    # Remove the last value and check the new mean and standard deviation.
    mystery_data.pop()
    numpy_data_modified = numpy.array(mystery_data)
    modified_mean = numpy_data_modified.mean()
    modified_std = numpy_data_modified.std()
    print("Second Mean = {0}\nSecond Std. Deviation = {1}"
          .format(modified_mean, modified_std))


def otsus_method(data):
    """Use Otsu's method for 1D clustering to separate the data into
    two groups.

    :param data: numpy.ndrray of floating point numbers
    :return: Threshold value as an integer. All values less-than or equal-to
    the threshold are in the first cluster and the remainder in the second.
    """
    # Quantize the data into bins of size 2 and recast them to ints.
    quantized_data = (numpy.floor(data / age_bin_size)
                      * age_bin_size).astype(int)

    # Try every non-redundant (i.e. even values only) and save the weighted
    # variances for comparison.
    all_weighted_variances = []
    for threshold in quantized_data:
        left_indices = (quantized_data <= threshold).nonzero()
        left_data = quantized_data.take(left_indices)
        weight_left = left_data.size/quantized_data.size
        variance_left = left_data.var()

        right_indices = (quantized_data > threshold).nonzero()
        right_data = quantized_data.take(right_indices)
        if right_data.size == 0:
            weight_right = 0
            variance_right = 0
        else:
            weight_right = right_data.size/quantized_data.size
            variance_right = right_data.var()

        all_weighted_variances.append(weight_left*variance_left +
                                      weight_right*variance_right)

    # Iterate through the data to find the minimum and print out any duplicates
    # found and ignored during the process.
    all_weighted_variances = numpy.array(all_weighted_variances)
    min_index = all_weighted_variances.argmin()
    min_variance = all_weighted_variances[min_index]
    for index in range(1, len(all_weighted_variances)):
        if all_weighted_variances[index] < min_variance:
            min_index = index
            min_variance = all_weighted_variances[index]
        elif all_weighted_variances[index] == min_variance:
            print("Duplicate found for age {0}.".format(quantized_data[index]))

    print("Threshold(s) for minimum variance: {0}".format(quantized_data[min_index]))
    print("Minimum variance = {0}".format(min_variance))

    return all_weighted_variances


def cost_function(data, alpha):
    """Combine the weighted variance metric from Otsu's method with a
    regularization value which gives more value to thresholds which more
    evenly divide the data.

    :param data: numpy.ndrray of real numbers
    :param alpha: factor to multiply the regularization by to change its weight
    :return: Threshold value as an integer. All values less-than or equal-to
    the threshold are in the first cluster and the remainder in the second.
    """
    # Quantize the data into bins of size 2 and recast them to ints.
    quantized_data = (numpy.floor(data / age_bin_size)
                      * age_bin_size).astype(int)

    # Try every non-redundant (i.e. even values only) and save the weighted
    # variances for comparison.
    all_weighted_variances = []
    for threshold in quantized_data:
        left_indices = (quantized_data <= threshold).nonzero()
        left_data = quantized_data.take(left_indices)
        weight_left = left_data.size/quantized_data.size
        variance_left = left_data.var()

        right_indices = (quantized_data > threshold).nonzero()
        right_data = quantized_data.take(right_indices)
        if right_data.size == 0:
            weight_right = 0
            variance_right = 0
        else:
            weight_right = right_data.size/quantized_data.size
            variance_right = right_data.var()

        # The value is the weighted variance of each side plus the
        # regularization factor. Norm-factor is constant at 100, but alpha
        # is a provided argument for the function.
        all_weighted_variances.append(weight_left*variance_left +
                                      weight_right*variance_right +
                                      numpy.abs(left_data.size -
                                                right_data.size)
                                      / norm_factor * alpha)

    # Iterate through the data to find the minimum and print out any duplicates
    # found and ignored during the process.
    min_variance = all_weighted_variances[0]
    min_index = 0
    for index in range(1, len(all_weighted_variances)):
        if all_weighted_variances[index] < min_variance:
            min_index = index
            min_variance = all_weighted_variances[index]

    print("Threshold(s) for minimum variance: {0}".format(quantized_data[min_index]))
    print("Minimum variance = {0}".format(min_variance))


def plot_mixed_variance(data):
    """Use Otsu's method to determine the threshold for clustering, then plot
    the data with their associated variance and add a circle around the point
    which had the minimum variance.

    :param data: numpy.ndarray of floating point numbers
    :return: None
    """
    # Quantize the data into bins of size 2 and recast them to ints.
    quantized_ages = (numpy.floor(data / age_bin_size)
                      * age_bin_size).astype(int)

    # Use Otsu's method without regularization to find mixed variance.
    all_weighted_variances = numpy.array(otsus_method(data))

    # Find the index of the minimum variance.
    min_index = all_weighted_variances.argmin()

    # Plot all ages and variances.
    pyplot.plot(quantized_ages, all_weighted_variances,
                   marker='d', markersize=2, linewidth=1)

    # Plot an extra point to highlight which age has the lowest variance.
    pyplot.scatter(quantized_ages[min_index], all_weighted_variances[min_index],
                marker='o', color='#fafa00', edgecolors='#ff0000')

    pyplot.title("Quantized Age vs Weighted Variance")
    pyplot.xlabel("Quantized Age (bin size = 2)")
    pyplot.ylabel("Weighted Variance (Otsu's Method)")
    pyplot.show()


def main():
    """Main function.
    Execute all other functions as necessary to complete parts 1-4 of HW01.
    """
    print("\nPart 1: Exploratory Data Analysis\n")
    mystery_data = read_mystery_data()
    mystery_exploratory_analysis(mystery_data)

    print("\nPart 2: Otsu\'s Method\n")
    abominable_data = read_abominable_data()
    otsus_method(abominable_data)

    print("\nPart 3: Cost Function\n")
    alpha_values = [100, 1, 0.2, 0.1, 0.05, 0.04, 0.02, 0.01, 0.001]
    for alpha in alpha_values:
        cost_function(abominable_data, alpha)

    print("\nPart 4: Graphing\n")
    plot_mixed_variance(abominable_data)


if __name__ == "__main__":
    main()
