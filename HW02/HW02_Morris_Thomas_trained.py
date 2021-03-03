"""
Author: Thomas Morris | tcm1998
Written 2020-02-09
"""

import numpy as np
import os.path
import sys
from matplotlib import pyplot as plt

AGE_BIN_SIZE = 2
HEIGHT_BIN_SIZE = 5


def read_data_file(data_file_name):
	"""Given a file name, locate, open, and read the file. The file is assumed
	to be a CSV file and its data will be parsed accordingly and returned as a
	numpy.ndarray.

	:param: data_file_name - <str> name of CSV file to read data from
	:return: <numpy.ndarray> all data read from data_file_name
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
	all_data = []
	with open(data_file_path, mode='r') as open_file:
		next(open_file)
		for line in open_file:
			line_data = line.strip().split(',')
			line_data[0] = int(np.floor(float(line_data[0]) / AGE_BIN_SIZE) *
						AGE_BIN_SIZE)
			line_data[1] = int(np.floor(float(line_data[1]) / HEIGHT_BIN_SIZE) *
						HEIGHT_BIN_SIZE)
			line_data[2] = int(line_data[2])
			all_data.append(line_data)

	return np.array(all_data)


def main(argv):
	"""Read the data from the provided file then attempt to classify it using
	a 1D binary sorting method. This function uses either height or age and
	uses a simple threshold test for sorting.

	:param data_file_arg: <str> Name of the CSV file of snowfolk data to
	classify.
	:return: None
	"""
	unclassified_data = read_data_file(argv[0])

	# Index 0 means we are classifying by age and 1 means we are using height.
	# Snowfolk under the first threshold are from Bhutan and the rest are
	# from Assam.
	for snowfolk in unclassified_data:
		if snowfolk[1] <= 135:
			print("+1")
		else:
			print("-1")


if __name__ == "__main__":
	main(sys.argv[1:])

