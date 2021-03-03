"""
Author: Thomas Morris | tcm1998
Date: 8 February 2020
"""

import numpy as np
import os.path
import sys
from matplotlib import pyplot as plt
from datetime import date

TRAINED_FILE_NAME = 'HW02_Morris_Thomas_Trained.py'
AGE_BIN_SIZE = 2
HEIGHT_BIN_SIZE = 5


def fwrite(string, target_file, tabs=0, newlines=1):
    """Add a number of tab characters to the front of a string."""
    target_file.write('\t'*tabs + string + '\n'*newlines)


def prolog(file_ptr):
    """Write the header docstring, import statements, and general function for
    extracting data from a CSV file of snowfolk data.

    :param file_ptr: open file to write code into
    :return: None
    """
    depth = 0
    # Write the header comment including date.
    fwrite("\"\"\"\nAuthor: Thomas Morris | tcm1998\nWritten {0}\n\"\"\""
           .format(date.today()), file_ptr, depth, newlines=2)

    # Add the import statements.
    fwrite("import numpy as np\nimport os.path\nimport sys\nfrom matplotlib"
           " import pyplot as plt", file_ptr, depth, newlines=2)

    # Bin size constants
    fwrite("AGE_BIN_SIZE = {0}\nHEIGHT_BIN_SIZE = {1}"
           .format(AGE_BIN_SIZE, HEIGHT_BIN_SIZE), file_ptr, depth, 3)

    # Finish with a function for reading a CSV file and returning its data.
    # Function def
    fwrite("def read_data_file(data_file_name):", file_ptr, depth)
    depth += 1

    # Docstring
    fwrite("\"\"\"Given a file name, locate, open, and read the file. The file"
           " is assumed", file_ptr, depth)
    fwrite("to be a CSV file and its data will be parsed accordingly and "
           "returned as a", file_ptr, depth)
    fwrite("numpy.ndarray.", file_ptr, depth, 2)
    fwrite(":param: data_file_name - <str> name of CSV file to read data from",
           file_ptr, depth)
    fwrite(":return: <numpy.ndarray> all data read from data_file_name",
           file_ptr, depth)
    fwrite("\"\"\"", file_ptr, depth)

    # Function body
    fwrite("# Assume the data file is in the current working directory.",
           file_ptr, depth)
    fwrite("cwd = os.getcwd()", file_ptr, depth)
    fwrite("data_file_path = os.path.join(os.getcwd(), data_file_name)",
           file_ptr, depth, 2)
    fwrite("# If it isn't present, check the parent folder.", file_ptr, depth)
    fwrite("if not os.path.exists(data_file_path):", file_ptr, depth)
    # Conditional
    depth += 1
    fwrite("super_folder = os.path.dirname(cwd)", file_ptr, depth)
    fwrite("data_file_path = os.path.join(super_folder, data_file_name)",
           file_ptr, depth, 2)
    fwrite("# If it still can't be found, exit.", file_ptr, depth)
    fwrite("if not os.path.exists(data_file_path):", file_ptr, depth)
    # Conditional
    depth += 1
    fwrite("sys.exit(\"Provided data file could not be found.\")", file_ptr,
           depth, 2)
    # Both conditionals end here with no else statements.
    depth -= 2
    fwrite("# We have found the file and can extract the data.", file_ptr,
           depth)
    fwrite("all_data = []", file_ptr, depth)
    fwrite("with open(data_file_path, mode='r') as open_file:", file_ptr, depth)
    # with body
    depth += 1
    fwrite("next(open_file)", file_ptr, depth)
    fwrite("for line in open_file:", file_ptr, depth)
    # Beginning of for loop
    depth += 1
    fwrite("line_data = line.strip().split(\',\')", file_ptr, depth)
    fwrite("line_data[0] = int(np.floor(float(line_data[0]) / AGE_BIN_SIZE) *",
           file_ptr, depth)
    fwrite("AGE_BIN_SIZE)", file_ptr, depth+3)
    fwrite("line_data[1] = int(np.floor(float(line_data[1]) / HEIGHT_BIN_SIZE)"
           " *", file_ptr, depth)
    fwrite("HEIGHT_BIN_SIZE)", file_ptr, depth+3)
    fwrite("line_data[2] = int(line_data[2])", file_ptr, depth)
    fwrite("all_data.append(line_data)", file_ptr, depth, 2)
    # End of for loop and with
    depth -= 2
    fwrite("return np.array(all_data)", file_ptr, depth, 3)
    # End of function
    depth -= 1


def write_body(file_ptr, attribute_idx, threshold, bhutan_first):
    """Write the main function of the trained program, incorporating the
    provided, predetermined attribute and threshold the program should classify
    snowfolk with.

    :param file_ptr: open file object to write the program into
    :param attribute_idx: <int> Index of the attribute from a snowfolk CSV file
        which will be tested against the threshold.
    :param threshold: Binary decision threshold for classification.
    :return: None
    """
    depth = 0
    # The body will only consist of the main function.
    # Function def
    fwrite("def main(argv):", file_ptr, depth)

    # Docstring
    depth += 1
    fwrite("\"\"\"Read the data from the provided file then attempt to "
           "classify it using", file_ptr, depth)
    fwrite("a 1D binary sorting method. This function uses either height or "
           "age and", file_ptr, depth)
    fwrite("uses a simple threshold test for sorting.", file_ptr, depth, 2)

    fwrite(":param data_file_arg: <str> Name of the CSV file of snowfolk data"
           " to", file_ptr, depth)
    fwrite("classify.", file_ptr, depth)
    fwrite(":return: None", file_ptr, depth)
    fwrite("\"\"\"", file_ptr, depth)
    fwrite("unclassified_data = read_data_file(argv[0])", file_ptr, depth, 2)
    fwrite("# Index 0 means we are classifying by age and 1 means we are using"
           " height.", file_ptr, depth)
    fwrite("# Snowfolk under the first threshold are from Bhutan and the rest"
           " are", file_ptr, depth)
    fwrite("# from Assam.", file_ptr, depth)
    fwrite("for snowfolk in unclassified_data:", file_ptr, depth)
    # For loop
    depth += 1
    fwrite("if snowfolk[{0}] <= {1}:".format(attribute_idx, threshold),
           file_ptr, depth)
    # Conditional
    if bhutan_first:
        fwrite("print(\"+1\")", file_ptr, tabs=depth+1)
        fwrite("else:", file_ptr, depth)
        fwrite("print(\"-1\")", file_ptr, tabs=depth+1, newlines=2)
    else:
        fwrite("print(\"-1\")", file_ptr, tabs=depth+1)
        fwrite("else:", file_ptr, depth)
        fwrite("print(\"+1\")", file_ptr, tabs=depth+1, newlines=2)
    # End Conditional
    # End of function
    depth -= 2


def epilog(file_ptr):
    """Write the conditional for main programs and close the file."""
    fwrite("\nif __name__ == \"__main__\":", file_ptr)
    fwrite("main(sys.argv[1:])", file_ptr, 1, 2)
    file_ptr.flush()
    file_ptr.close()


def build_classifier(file_name):
    """Use the complete data in the provided CSV file to determine how best
    to classify abominable snowfolk. Determine the best attribute, threshold
    value, and which class falls on which side of the threshold, and return
    all of this information.

    :param file_name: <str> CSV file of training data
    :return: (<int>, <int>, <int>) attribute index (0 or 1), threshold value,
        and a boolean (1 or 0) specifying if Bhutan snowfolk are below the
        threshold (if 1) or Assam are below the threshold (if 0).
    """
    # Assume the data file is in the current working directory.
    cwd = os.getcwd()
    data_file_path = os.path.join(os.getcwd(), file_name)

    # If it isn't present, check the parent folder.
    if not os.path.exists(data_file_path):
        super_folder = os.path.dirname(cwd)
        data_file_path = os.path.join(super_folder, file_name)

        # If it still can't be found, exit.
        if not os.path.exists(data_file_path):
            sys.exit("Provided data file could not be found.")

    sample_data = []
    with open(data_file_path, 'r') as open_file:
        next(open_file)
        for line in open_file:
            line_data = line.strip().split(',')
            line_data[0] = int(np.floor(float(line_data[0])/AGE_BIN_SIZE) *
                               AGE_BIN_SIZE)
            line_data[1] = int(np.floor(float(line_data[1])/HEIGHT_BIN_SIZE) *
                               HEIGHT_BIN_SIZE)
            line_data[2] = int(line_data[2])
            sample_data.append(line_data)

    # Convert it to a numpy array.
    sample_data = np.array(sample_data)

    plt.figure(figsize=(12, 6))

    age_thresh, age_mistakes, age_class_order = \
        find_best_threshold(sample_data, 0)
    height_thresh, height_mistakes, height_class_order = \
        find_best_threshold(sample_data, 1)

    plt.show()

    if age_mistakes < height_mistakes:
        return 0, age_thresh, age_class_order
    else:
        return 1, height_thresh, height_class_order


def find_best_threshold(sample_data, attribute):
    """Generic method of testing all reasonable thresholds for one attribute
    and determining the best threshold and the class relationships to the
    threshold. This is done by using the index of the attribute for all
    operations.

    :param sample_data: <numpy.ndarray> all training data
    :param attribute: <int> index of the attribute to find a threshold for
    :return: (<int>, <int>, <int>) threshold, number of mistakes, and whether
        Bhutans are expected to be below the threshold or above
    """
    # Determine which constant should be used for this attribute.
    if attribute == 0:
        bin_size = AGE_BIN_SIZE
    else:
        bin_size = HEIGHT_BIN_SIZE

    mistakes_bhutan = []
    mistakes_assam = []

    minimums = sample_data.min(0)
    maximums = sample_data.max(0)

    # Find the best age threshold.
    best_threshold = minimums[0]
    min_mistakes = sample_data.size

    # This is for whether the Bhutans should be below the threshold.
    bhutan_prime = 1

    # For all reasonable age thresholds.
    for threshold in range(minimums[attribute], maximums[attribute] + bin_size,
                           bin_size):

        # Bhutan less than Assam classification mistakes
        blt_mistakes = 0

        # Assam less than Bhutan classification mistakes
        alt_mistakes = 0

        for values in sample_data:
            # If the snowfolk is below the threshold and is from the Assam
            # region, then this is a mistake for attr <= threshold := B
            if (values[attribute] <= threshold and values[2] < 0) or \
                    (values[attribute] > threshold and values[2] > 0):
                blt_mistakes += 1

            # If the snowfolk is younger than the threshold and is from the
            # Bhutan region, then this is a mistake for age <= threshold := A
            if (values[attribute] <= threshold and values[2] > 0) or \
                    (values[attribute] > threshold and values[2] < 0):
                alt_mistakes += 1

        # Record mistake totals for graphing.
        mistakes_assam.append(alt_mistakes)
        mistakes_bhutan.append(blt_mistakes)

        # If the rule attr <= threshold := Bhutan is better than the existing
        # rule, make the following updates.
        if blt_mistakes < min_mistakes:
            best_threshold = threshold
            min_mistakes = blt_mistakes
            bhutan_prime = 1

        # If the rule attr <= threshold := Assam is better than the existing
        # rule, make the following updates.
        if alt_mistakes < min_mistakes:
            best_threshold = threshold
            min_mistakes = alt_mistakes
            bhutan_prime = 0

    # Plot mistakes against the range of thresholds tested.
    x = range(minimums[attribute], maximums[attribute]+bin_size, bin_size)
    if attribute == 0:
        plt.subplot(121)
        plt.xlabel('Age Threshold (years)')
        legend = ['Assam <= Threshold', 'Bhutan <= Threshold']
        plt.title('Mistakes for Age Thresholds')
    else:
        plt.subplot(122)
        plt.xlabel('Height Threshold (cm)')
        legend = ['Assam <= Threshold', 'Bhutan <= Threshold']
        plt.title('Mistakes for Height Thresholds')
    plt.plot(x, mistakes_assam, x, mistakes_bhutan)
    plt.ylabel('Mistakes')
    plt.legend(legend)
    return best_threshold, min_mistakes, bhutan_prime


def roc_curve(file_name):
    """Graph a receiver-operator curve for the age and height thresholds, with
    the knowledge of which class is expected on which side of the thresholds
    gained from already completing the assignment.

    :param file_name: <str> CSV file of training data
    :return: (<int>, <int>, <int>) attribute index (0 or 1), threshold value,
        and a boolean (1 or 0) specifying if Bhutan snowfolk are below the
        threshold (if 1) or Assam are below the threshold (if 0).
    """
    # Assume the data file is in the current working directory.
    cwd = os.getcwd()
    data_file_path = os.path.join(os.getcwd(), file_name)

    # If it isn't present, check the parent folder.
    if not os.path.exists(data_file_path):
        super_folder = os.path.dirname(cwd)
        data_file_path = os.path.join(super_folder, file_name)

        # If it still can't be found, exit.
        if not os.path.exists(data_file_path):
            sys.exit("Provided data file could not be found.")

    sample_data = []
    # Quantize the data.
    with open(data_file_path, 'r') as open_file:
        next(open_file)
        for line in open_file:
            line_data = line.strip().split(',')
            line_data[0] = int(np.floor(float(line_data[0])/AGE_BIN_SIZE) *
                               AGE_BIN_SIZE)
            line_data[1] = int(np.floor(float(line_data[1])/HEIGHT_BIN_SIZE) *
                               HEIGHT_BIN_SIZE)
            line_data[2] = int(line_data[2])
            sample_data.append(line_data)

    # Convert it to a numpy array.
    sample_data = np.array(sample_data)
    minimums = sample_data.min(0)
    maximums = sample_data.max(0)

    # Set up lists for the data.
    assam_age_true_positives = []
    assam_age_false_alarms = []
    bhutan_height_true_positives = []
    bhutan_height_false_alarms = []

    # Test all age thresholds.
    best_far_age = 0
    best_tpr_age = 0
    min_mistakes = len(sample_data)
    for age_threshold in range(minimums[0], maximums[0]+AGE_BIN_SIZE,
                               AGE_BIN_SIZE):
        true_positives = 0
        false_alarms = 0
        misses = 0
        for values in sample_data:
            if values[0] <= age_threshold:
                if values[2] > 0:
                    false_alarms += 1
                else:
                    true_positives += 1
            elif values[0] > age_threshold and values[2] < 0:
                misses += 1

        assam_age_true_positives.append(true_positives)
        assam_age_false_alarms.append(false_alarms)

        # Check if this is the best age threshold.
        if false_alarms + misses < min_mistakes:
            min_mistakes = false_alarms + misses
            best_far_age = false_alarms/len(sample_data)
            best_tpr_age = true_positives/len(sample_data)

    # Test all height thresholds.
    best_far_height = 0
    best_tpr_height = 0
    min_mistakes = len(sample_data)
    for height_threshold in range(minimums[1], maximums[1]+HEIGHT_BIN_SIZE,
                                  HEIGHT_BIN_SIZE):
        true_positives = 0
        false_alarms = 0
        misses = 0
        for values in sample_data:
            if values[1] <= height_threshold:
                if values[2] < 0:
                    false_alarms += 1
                else:
                    true_positives += 1
            elif values[1] > height_threshold and values[2] > 0:
                misses += 1

        bhutan_height_true_positives.append(true_positives)
        bhutan_height_false_alarms.append(false_alarms)

        # Check if this is the best height threshold.
        if false_alarms + misses < min_mistakes:
            min_mistakes = false_alarms + misses
            best_far_height = false_alarms/len(sample_data)
            best_tpr_height = true_positives/len(sample_data)

    # Normalize the true and false positives by the size of the data.
    assam_age_true_positives = np.array(assam_age_true_positives)/len(sample_data)
    assam_age_false_alarms = np.array(assam_age_false_alarms)/len(sample_data)
    bhutan_height_true_positives = np.array(bhutan_height_true_positives)/len(sample_data)
    bhutan_height_false_alarms = np.array(bhutan_height_false_alarms)/len(sample_data)

    # Plot the curve with labeled axes and a legend.
    plt.figure(figsize=(9, 9))
    plt.plot(assam_age_false_alarms, assam_age_true_positives, 'r',
             bhutan_height_false_alarms, bhutan_height_true_positives, 'g',
             best_far_age, best_tpr_age, 'bo',
             best_far_height, best_tpr_height, 'yo')
    plt.xlabel("False Alarm Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Snowfolk Classification ROC Curve")
    plt.legend(['Age (A < B)', 'Height (B < A)',
                'Age Threshold with Fewest Mistakes',
                'Height Threshold with Fewest Mistakes'], loc=4)
    plt.show()


def main(training_file_name):
    """Main function.
    Determine the best classification attribute and threshold, then write the
    training program using this information.

    :param training_file_name: <str> name of the training data file
    :return: None
    """
    attribute, threshold, class_order = build_classifier(training_file_name)
    trained_file = open(TRAINED_FILE_NAME, mode='w')
    prolog(trained_file)
    write_body(trained_file, attribute, threshold, class_order)
    epilog(trained_file)

    roc_curve(training_file_name)


if __name__ == "__main__":
    main(sys.argv[1])
