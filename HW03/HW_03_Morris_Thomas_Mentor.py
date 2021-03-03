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
           " import pyplot as plt\nimport pandas", file_ptr, depth, newlines=3)

    # Finish with a function for reading a CSV file and returning its data.
    # Function def
    fwrite("def read_data_file(data_file_name):", file_ptr, depth)
    depth += 1

    # Docstring
    fwrite("\"\"\"Given a file name, locate, open, and read the file. The file"
           " is assumed", file_ptr, depth)
    fwrite("to be a CSV file and its data will be parsed accordingly and "
           "returned as a", file_ptr, depth)
    fwrite("pandas dataframe.", file_ptr, depth, 2)
    fwrite(":param: data_file_name - <str> name of CSV file to read data from",
           file_ptr, depth)
    fwrite(":return: <Pandas.DataFrame> all data read from data_file_name",
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
    fwrite("return pandas.read_csv(data_file_path, delimiter=',')",
           file_ptr, depth, newlines=3)


def write_body(file_ptr, attribute, inverse_relation):
    """Write the main function of the trained program, incorporating the
    provided, predetermined attribute in order to guess which people will be
    sick using a One Rule.

    :param file_ptr: open file object to write the program into
    :param attribute: <int> Name of the attribute from the CSV file which is
        the subject of our One Rule.
    :param inverse_relation: <bool> True if there is an inverse relationship,
        meaning that when the the food is not eaten a person will become sick.
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

    fwrite(":param data_file_arg: <str> Name of the CSV file of CDC data to",
           file_ptr, depth)
    fwrite("make predictions for.", file_ptr, depth)
    fwrite(":return: None", file_ptr, depth)
    fwrite("\"\"\"", file_ptr, depth)
    fwrite("testing_data = read_data_file(argv[0])", file_ptr, depth, 2)
    fwrite("relevant_data = testing_data[[\"{0}\"]]".format(attribute),
           file_ptr, depth)
    fwrite("for row in relevant_data.itertuples(index=False):",
           file_ptr, depth)
    # For loop
    depth += 1
    fwrite("if row.{0} > 0:".format(attribute), file_ptr, depth)
    # Conditional
    if not inverse_relation:
        fwrite("print(\"1\")", file_ptr, tabs=depth+1)
        fwrite("else:", file_ptr, depth)
        fwrite("print(\"0\")", file_ptr, tabs=depth+1, newlines=2)
    else:
        fwrite("print(\"0\")", file_ptr, tabs=depth+1)
        fwrite("else:", file_ptr, depth)
        fwrite("print(\"1\")", file_ptr, tabs=depth+1, newlines=2)
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
    training_data = pandas.read_csv(data_file_path, delimiter=',')

    # Delete the GuestID column, as it serves no purpose here.
    del training_data['GuestID']

    # Calculate the cross-correlational coefficients.
    cross_correlation_matrix = training_data.corr()

    # Print the cross-correlational matrix.
    # print(cross_correlation_matrix)

    return find_one_rule(training_data)


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
    attribute, inverse = build_classifier(training_file_name)
    trained_file = open(TRAINED_FILE_NAME, mode='w')
    prolog(trained_file)
    write_body(trained_file, attribute, inverse)
    epilog(trained_file)


if __name__ == "__main__":
    main(sys.argv[1])
