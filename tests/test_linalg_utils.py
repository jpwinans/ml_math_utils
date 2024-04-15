import sys
import os

# Add the root directory to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from linalg_utils import *


# Tests for swap_rows function
def test_swap_rows_valid_indices():
    """
    Test case with valid row indices.
    """
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    row_index_1 = 0
    row_index_2 = 2

    expected_result = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])

    result = swap_rows(M, row_index_1, row_index_2)
    assert np.array_equal(result, expected_result)


def test_swap_rows_same_index():
    """
    Test case with the same row index for both indices.
    """
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    row_index_1 = 1
    row_index_2 = 1

    expected_result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = swap_rows(M, row_index_1, row_index_2)
    assert np.array_equal(result, expected_result)


def test_get_index_first_non_zero_value_from_column_non_zero_value_exists():
    """
    Test case where a non-zero value exists in the specified column.
    """
    M = np.array([[0, 2, 0], [0, 0, 3], [4, 0, 0]])
    column = 1
    starting_row = 0

    expected_index = 0

    result = get_index_first_non_zero_value_from_column(M, column, starting_row)
    assert result == expected_index


def test_get_index_first_non_zero_value_from_column_no_non_zero_value():
    """
    Test case where no non-zero value exists in the specified column.
    """
    M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    column = 1
    starting_row = 0

    expected_index = -1

    result = get_index_first_non_zero_value_from_column(M, column, starting_row)
    assert result == expected_index


def test_get_index_first_non_zero_value_from_row_non_zero_value_exists():
    """
    Test case where a non-zero value exists in the specified row.
    """
    M = np.array([[0, 2, 0], [0, 0, 3], [4, 0, 0]])
    row = 1

    expected_index = 2

    result = get_index_first_non_zero_value_from_row(M, row)
    assert result == expected_index


def test_get_index_first_non_zero_value_from_row_augmented_matrix():
    """
    Test case with an augmented matrix where the last column is ignored.
    """
    M = np.array([[0, 2, 0, 5], [0, 0, 0, 7], [4, 0, 0, 9]])
    row = 1
    augmented = True

    expected_index = -1

    result = get_index_first_non_zero_value_from_row(M, row, augmented)
    assert result == expected_index

    M = np.array([[0, 2, 0, 5], [0, 0, 4, 7], [4, 0, 0, 9]])
    row = 1
    augmented = True

    expected_index = 2

    result = get_index_first_non_zero_value_from_row(M, row, augmented)
    assert result == expected_index


def test_augmented_matrix():
    """
    Test case for creating an augmented matrix by horizontally stacking two matrices.
    """
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7], [8]])

    expected_augmented_matrix = np.array([[1, 2, 3, 7], [4, 5, 6, 8]])

    result = augmented_matrix(A, B)
    assert np.array_equal(result, expected_augmented_matrix)


def test_row_echelon_form_non_singular_matrix():
    """
    Test case with a non-singular matrix.
    """
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    B = np.array([[8], [-11], [-3]])

    expected_result = np.array([[1, 0.5, -0.5, 4], [0, 1, 1, 2], [0, 0, 1, -1]])

    result = row_echelon_form(A, B)
    print(result)
    assert np.allclose(result, expected_result)


def test_row_echelon_form_singular_matrix():
    """
    Test case with a singular matrix.
    """
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1], [2], [3]])

    expected_result = "Singular system"
    
    print(row_echelon_form(A, B))

    result = row_echelon_form(A, B)
    assert result == expected_result


def test_back_substitution_unique_solution():
    """
    Test case with an augmented matrix in row echelon form with a unique solution.
    """
    M = np.array([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, -1]])

    expected_solution = np.array([2, 3, -1])

    result = back_substitution(M)
    assert np.allclose(result, expected_solution)


def test_back_substitution_another_unique_solution():
    """
    Test case with another augmented matrix in row echelon form with a unique solution.
    """
    M = np.array([[1, 0, 0, 0, 2], [0, 1, 0, 0, -1], [0, 0, 1, 0, 3], [0, 0, 0, 1, 4]])

    expected_solution = np.array([2, -1, 3, 4])

    result = back_substitution(M)
    assert np.allclose(result, expected_solution)


def test_gaussian_elimination_unique_solution():
    """
    Test case with a linear system that has a unique solution.
    """
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    B = np.array([[8], [-11], [-3]])

    expected_solution = np.array([2, 3, -1])

    result = gaussian_elimination(A, B)
    assert np.allclose(result, expected_solution)


def test_gaussian_elimination_singular_system():
    """
    Test case with a singular linear system.
    """
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1], [2], [3]])

    expected_result = "Singular system"

    result = gaussian_elimination(A, B)
    assert result == expected_result


# Run the tests with pytest tests/test_linalg_utils.py
