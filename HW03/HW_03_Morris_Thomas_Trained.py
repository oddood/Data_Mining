"""
Author: Thomas Morris | tcm1998
Written 2020-02-15
"""

import numpy as np
import os.path
import sys
from matplotlib import pyplot as plt
import pandas


def read_data_file(data_file_name):
	"""Given a file name, locate, open, and read the file. The file is assumed
	to be a CSV file and its data will be parsed accordingly and returned as a
	pandas dataframe.

	:param: data_file_name - <str> name of CSV file to read data from
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
	return pandas.read_csv(data_file_path, delimiter=',')


def main(argv):
	"""Read the data from the provided file then attempt to classify it using
	a 1D binary sorting method. This function uses either height or age and
	uses a simple threshold test for sorting.

	:param data_file_arg: <str> Name of the CSV file of CDC data to
	make predictions for.
	:return: None
	"""
	testing_data = read_data_file(argv[0])

	relevant_data = testing_data[["PeanutButter"]]
	for row in relevant_data.itertuples(index=False):
		if row.PeanutButter > 0:
			print("1")
		else:
			print("0")


if __name__ == "__main__":
	main(sys.argv[1:])

