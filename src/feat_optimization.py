from sklearn.metrics import adjusted_rand_score
import typing as t
import numpy as np


def get_rand_index(clustering_results: t.List[t.List[int]]):
    # Calculate similarity scores using Adjusted Rand Index
    similarity_matrix = np.zeros((len(clustering_results), len(clustering_results)))
    for i in range(len(clustering_results)):
        for j in range(i + 1, len(clustering_results)):
            score = adjusted_rand_score(clustering_results[i], clustering_results[j])
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score
    np.fill_diagonal(similarity_matrix, 1)

    return similarity_matrix


def get_mean(similarity_matrix):
    # Create a mask to exclude the diagonal elements
    mask = np.eye(similarity_matrix.shape[0], dtype=bool)

    # Apply the mask to the similarity matrix
    masked_matrix = np.ma.masked_array(similarity_matrix, mask)

    # Calculate the mean excluding the diagonal elements
    average_value = np.mean(masked_matrix)
    return average_value
