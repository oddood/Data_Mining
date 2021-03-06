"""
Program Name: HW07_Vo_Kelly_Morris_Thomas.py
Author: Kelly Vo <kdv7923>
Author: Thomas Morris <tcm1998>

This program's goal is to compute the cross-correlation coefficient of all
attributes, then cluster the data by agglomeration, using Euclidean distance
for calculating linkage.
"""
import os
import csv
import sys
import pandas as pd
import numpy as np


def compute_cross_correlation(csv_file):
    """
    This function computes all cross correlation of each attribute.

    :param csv_file: is the csv file
    """
    # Read in the data from the CSV file
    read_file = pd.read_csv(csv_file, encoding='utf-8')
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    csv_title = rows[0][1:]
    # csv_title = csv_title[1:]
    for i in range(len(csv_title)):
        # List if sorted cross-correlations
        sorted_lst = []
        # List of the absolute values of the matrix
        abs_lst = []
        # Computes the cross-correlation of each attribute
        for j in range(len(csv_title)):
            sorted_lst.append(read_file[csv_title[i]].corr(read_file[csv_title[j]]))
            abs_lst.append(np.abs(read_file[csv_title[i]].corr(read_file[csv_title[j]])))

        print()
        # Prints the attribute
        print(str(csv_title[i]).strip())
        # Prints the average of the absolute value of the matrix
        print("Mean of the absolute value of the matrix: " + str(np.average(abs_lst)))
        # Computes the most correlated attributes
        for j in range(len(csv_title)):
            if np.abs(sorted(sorted_lst)[len(sorted_lst) - 2]) > np.abs(sorted(sorted_lst)[0]):
                most_correlated = sorted(sorted_lst)[len(sorted_lst) - 2]
            else:
                most_correlated = sorted(sorted_lst)[0]
            if most_correlated == (read_file[csv_title[i]].corr(read_file[csv_title[j]])):
                # Prints the most strongly correlated
                print("Most strongly correlated: " + str(csv_title[i]).strip()
                      + " and " + str(csv_title[j]).strip() + " = " + str(
                    read_file[csv_title[i]].corr(read_file[csv_title[j]])))


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
    data = pd.read_csv(data_file_path, delimiter=',', index_col='ID')

    # Cast everything to integer.
    return data.astype(int)


def calculate_linkage(center1, center2):
    """
    Calculate the distance between two cluster centroids using Euclidean
    distance.
    :param center1: <Pandas.Series> Centroid of first cluster.
    :param center2: <Pandas.Series> Centroid of second cluster.
    :return: <float> Distance between center1 and center2.
    """
    # Convert the Series to numpy.ndarray for calculations.
    array1 = center1.to_numpy(dtype=float, copy=True)
    array2 = center2.to_numpy(dtype=float, copy=True)

    # Square root of the sum of squared differences.
    difference = array1 - array2
    squared = difference * difference
    squared_sum = squared.sum()
    return np.sqrt(squared_sum)


def initialize(dataframe):
    """
    Create a symmetric n-by-n matrix of floats, holding the distance from every
    record to every other record, which is to say, every cluster to every other
    cluster. The cluster's index in the matrix is its row Index in the original
    DataFrame.
    Also, create a dictionary to hold all the records of a cluster. The key is
    the cluster ID, which is its records' lowest Index, and the value is a
    DataFrame of all the associated records.

    :param dataframe: <Pandas.DataFrame> original table to divide into clusters
    :return: <List[List[float]], Dict> Distances matrix and cluster dictionary
    """
    row_count = dataframe.shape[0]

    # This n-by-n matrix holds the distance from each prototype to every other
    # prototype. It is a symmetric matrix with zeroes on the main diagonal.
    distances = [[0 for _ in range(row_count+1)] for _ in range(row_count+1)]

    # Clusters are held in a dictionary where the key is the ID, the row Index
    # of the record, and the value is a DataFrame.
    clusters = {}

    # For each record, make a new DataFrame of only that record and put it in
    # the clusters dictionary, then populate the distances matrix.
    for idx, record in dataframe.iterrows():
        clusters[idx] = pd.DataFrame([record])

        for idx2, record2 in dataframe.iterrows():
            if idx2 > idx:
                linkage = calculate_linkage(record, record2)
                distances[idx][idx2] = linkage
                distances[idx2][idx] = linkage

    return distances, clusters


def agglomeration(distances, clusters, centroids):
    """
    Continuously combine the two closest clusters until all records are
    combined. At 6 groups left, report some data out through the console.

    Note: when clusters are merged, the name of a cluster, and thus its
        centroid's row ID in the centroids table is the lowest row ID of all
        rows in the cluster, as it appeared in the original CSV file.

    :param distances: <List [List [int]]> Indexes match the row ID of a cluster;
        location i, j is the distance between clusters/centroids i and j. The
        matrix is symmetric.
    :param clusters: <dict> Dictionary where keys are row IDs and values are
        Pandas.DataFrame objects containing all records in that cluster.
    :param centroids: <Pandas.DataFrame> A set of records which represent the
        center of each cluster. No records are deleted from it since the
        records considered during agglomeration are based on the keys in
        clusters.
    :return: None
    """
    merged_cluster_size = []

    while len(clusters) > 1:
        min_distance = sys.float_info.max
        master_index = 0
        slave_index = 0

        # Check every remaining cluster against every other remaining cluster
        # using record IDs and find the two closets to each other.
        for start in clusters.keys():
            for end in clusters.keys():

                # We don't want to double-check values and we want start to be
                # smaller than end for consistent merging.
                if start >= end:
                    continue

                # If the distance between these clusters is smaller than our
                # current minimum, save the index values and distance.
                if distances[start][end] < min_distance:
                    min_distance = distances[start][end]
                    master_index = start
                    slave_index = end

        # Merge the two clusters by combining their record sets.
        # The master index is the smaller number and represents the cluster
        # which agglomerates records in the merge.
        clusters[master_index] = clusters[master_index]\
            .append(clusters[slave_index], ignore_index=True)

        # Save the size of the slave cluster for analysis.
        merged_cluster_size.append(clusters[slave_index].shape[0])

        # Remove the slave cluster.
        del clusters[slave_index]

        # Calculate the average of each column in the expanded table.
        new_averages = clusters[master_index].mean(axis=0)

        # Update each column of the master's centroid with its new average.
        for column in centroids.columns:
            centroids.loc[master_index, column] = new_averages[column]

        # Update the distances of the master cluster to all remaining clusters.
        for cluster_idx in clusters.keys():
            linkage = calculate_linkage(centroids.loc[cluster_idx],
                                        centroids.loc[master_index])
            distances[master_index][cluster_idx] = linkage
            distances[cluster_idx][master_index] = linkage

        # When we get down to six clusters, do some special things.
        if len(clusters) == 6:
            for k, v in clusters.items():
                print(v.shape[0])
                print(centroids.loc[k])

    # After agglomeration, report the last 18 sizes of merged clusters.
    print(merged_cluster_size[len(merged_cluster_size)-18:])


def main():
    """Read CSV file name from the command line arguments, then analyze and
    cluster the data.
    """
    # Check for proper input.
    try:
        sys.argv[1]
    except IndexError:
        print('Please input csv file')
        exit(0)

    # Make printing a table this big a bit cleaner.
    pd.set_option('display.max_rows', 25)
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 110)

    data = None
    # Read the data file.
    try:
        data = read_data_file(sys.argv[1])
    except FileNotFoundError:
        print('File not found.')
        exit(0)

    # Calculate and print the cross-correlational coefficients.
    cross_correlation = data.corr()
    print(cross_correlation)

    # Computes cross correlation of all attributes
    compute_cross_correlation(sys.argv[1])

    # Initialize our size 1 clusters and the distance matrix.
    dist_matrix, clusters = initialize(data)

    # Agglomeration clustering
    agglomeration(dist_matrix, clusters, data.copy().astype(float))


if __name__ == '__main__':
    main()
