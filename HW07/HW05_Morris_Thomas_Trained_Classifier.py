"""
Author: Thomas Morris | tcm1998
Written 2020-03-13
"""

import numpy as np
import os.path
import sys
from matplotlib import pyplot as plt
import pandas
from math import floor

AGE_BIN_SIZE = 2
HEIGHT_BIN_SIZE = 5


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
	data = pandas.read_csv(data_file_path, delimiter=',')

	# Round TailLn, HairLn, BangLn, Reach
	clean_data = data[['TailLn', 'HairLn', 'BangLn', 'Reach']].round(decimals=0)

	# Quantize the age by their respective bin sizes and save the new columns
	# Into the clean data frame.
	clean_data['Age'] = data['Age'].apply(lambda x: floor(x / AGE_BIN_SIZE) * AGE_BIN_SIZE)
	clean_data['Ht'] = data['Ht'].apply(lambda x: floor(x / HEIGHT_BIN_SIZE) * HEIGHT_BIN_SIZE)
	clean_data['Class'] = data['Class']

	# Cast everything to integer.
	return clean_data.astype(int)


def main(data_file_name):
	"""Read the data from the provided file then attempt to classify it using
	a decision tree. Each decision node makes a binary split on an integer threshold.

	:param data_file_name: <str> Name of the CSV file of CDC data to
	make predictions for.
	:return: None
	"""
	testing_data = read_data_file(data_file_name)

	for row in testing_data.itertuples(index=False):
		if row.BangLn >= 6:
			if row.Age >= 47:
				if row.Ht >= 141:
					print('-1')
				else:
					print('+1')
			else:
				if row.Ht >= 126:
					print('-1')
				else:
					print('+1')
		else:
			if row.Age >= 41:
				if row.Reach >= 176:
					print('-1')
				else:
					print('+1')
			else:
				if row.Ht >= 146:
					print('-1')
				else:
					print('+1')
		

if __name__ == "__main__":
	main(sys.argv[1])

