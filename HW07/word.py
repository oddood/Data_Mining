import docx
import pandas as pd
import os
import sys


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
    data = pd.read_csv(data_file_path, delimiter=',')

    # Cast everything to integer.
    # return data.astype(int)
    return data.replace({'Class': {'Assam': -1, 'Bhuttan': 1}})


# i am not sure how you are getting your data, but you said it is a
# pandas data frame
dat = read_data_file('HW_PCA_SHOPPING_CART_v895.csv')
df = dat.corr()
df = df.round(decimals=3)

# open an existing document
doc = docx.Document()

# add a table to the end and create a reference variable
# extra row is so we can add the header row
t = doc.add_table(df.shape[0]+1, df.shape[1])

# add the header rows.
for j in range(df.shape[-1]):
    t.cell(0,j).text = df.columns[j]

# add the rest of the data frame
for i in range(df.shape[0]):
    for j in range(df.shape[-1]):
        t.cell(i+1,j).text = str(df.values[i,j])

# save the doc
doc.save('Corr.docx')