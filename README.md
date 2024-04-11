# ml_math_utils
A collection of linear algebra, calculus, and statistics utilities useful in some machine learning applications. Current main use is to use gaussian_elimination to solve a system of equations with a unique solution.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jpwinans/ml_math_utils.git
   ```

2. Navigate to the project directory:
   ```
   cd ml_math_utils
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The `ml_math_utils` package provides various utility functions for linear algebra, calculus, and statistics. Here are some examples of how to use the functions:

```python
import numpy as np
from linalg_utils import swap_rows, get_index_first_non_zero_value_from_column, get_index_first_non_zero_value_from_row, row_echelon_form, back_substitution, gaussian_elimination

# Example usage of gaussian_elimination
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
B = np.array([[8],
              [-11],
              [-3]])
solution = gaussian_elimination(A, B)

# Example usage of back_substitution
M = np.array([[1, 0, 0, 2],
              [0, 1, 0, 3],
              [0, 0, 1, -1]])
solution = back_substitution(M)

# Example usage of row_echelon_form
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
B = np.array([[8],
              [-11],
              [-3]])
result = row_echelon_form(A, B)

# Example usage of swap_rows
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
row_index_1 = 0
row_index_2 = 2
result = swap_rows(M, row_index_1, row_index_2)

# Example usage of get_index_first_non_zero_value_from_column
M = np.array([[0, 2, 0],
              [0, 0, 3],
              [4, 0, 0]])
column = 1
starting_row = 0
index = get_index_first_non_zero_value_from_column(M, column, starting_row)

# Example usage of get_index_first_non_zero_value_from_row
M = np.array([[0, 2, 0],
              [0, 0, 3],
              [4, 0, 0]])
row = 1
index = get_index_first_non_zero_value_from_row(M, row)
```

## Testing

The `ml_math_utils` package includes a test suite to ensure the correctness of the implemented functions. To run the tests, follow these steps:

1. Make sure you have the required dependencies installed (see the Installation section).

2. Navigate to the project directory:
   ```
   cd ml_math_utils
   ```

3. Run the tests using pytest:
   ```
   pytest tests/
   ```

The test suite includes test cases for various functions in the `linalg_utils` module, such as `swap_rows`, `get_index_first_non_zero_value_from_column`, `get_index_first_non_zero_value_from_row`, `row_echelon_form`, `back_substitution`, and `gaussian_elimination`.

If all the tests pass successfully, you should see output indicating that the tests ran without any failures.

## Contributing

Contributions to the `ml_math_utils` package are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).