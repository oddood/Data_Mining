# DataFrame of feature averages for each class.
    class_feature_means_table = pandas.DataFrame(columns=['Assam', 'Bhuttan'])

    # # Populate the averages table.
    # for group, rows in data.groupby(['Class']):
    #    class_feature_means[group] = rows.mean()

    # Accumulated intra-class scatter matrices.
    within_class_scatter_matrix = np.zeros((num_features, num_features))

    for group, rows in data.groupby(['Class']):
        # Drop the classification for these rows.
        rows = rows.drop(['Class'], axis=1)

        # Calculate the means and store them in a separate table.
        class_feature_means = rows.mean()
        class_feature_means_table[group] = class_feature_means

        # Intra-class scatter matrix.
        scatter_matrix = np.zeros((num_features, num_features))

        for index, row in rows.iterrows():
            # Reshape these Series from one row many columns to one column with
            # many rows for matrix multiplication in the next step.
            vert_row, vert_feature_means = row.values.reshape(num_features, 1),\
                    class_feature_means.values.reshape(num_features, 1)

            # Row minus average dot product with its transpose, added to the
            # intra-class scatter matrix.
            scatter_matrix += (vert_row - vert_feature_means).\
                dot((vert_row - vert_feature_means).T)

        # Now that the scatter matrix is populated, add its contents to
        # the matrix which accumulates intra-class scatter matrices.
        within_class_scatter_matrix += scatter_matrix

    # Feature means for ALL data.
    feature_means = data.mean()
    between_class_scatter_matrix = np.zeros((num_features, num_features))

    for group in class_feature_means_table:
        # Weird method of finding how many instances of the class exist.
        class_frequency = len(data.loc[data['Class'] == group].index)

        # Reshape the feature means for this class and the means for all data.
        class_means, means = class_feature_means_table[group].\
            values.reshape(num_features, 1),\
            feature_means.values.reshape(num_features, 1)

        # Accumulate the class frequencies multiplied by dot product of the
        # class means minus global means and its transpose.
        between_class_scatter_matrix += \
            class_frequency * (class_means - means).dot((class_means - means).T)

    # Now we obtain all Eigenvalues and Eigenvectors.
    eigen_values, eigen_vectors = np.linalg.eig(
        np.linalg.inv(within_class_scatter_matrix).
        dot(between_class_scatter_matrix)
    )

    # Combine the Eigenvalues and vectors.
    pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
             for i in range(len(eigen_values))]

    # TODO DO THE NESTED LOOP HERE??
    # TODO
    w_matrix = np.hstack((pairs[0][1].reshape(13, 1),
                          pairs[1][1].reshape(num_features, 1))).real

    # TODO
    data = data.replace({'Class': {'Assam': 0, 'Bhuttan': 1}})
    y_lda = data['Class']
    stripped_data = data.drop('Class')
    x_lda = np.array(stripped_data.dot(w_matrix))

    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(stripped_data, y_lda)