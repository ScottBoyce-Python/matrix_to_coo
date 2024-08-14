"""
Developed by Scott E. Boyce
Boyce@engineer.com

This module provides a function to convert a dense numpy matrix to a Coordinate list (COO) format.
The COO format stores a list of (row, column, value) for non-zero values in a matrix,
similar to scipy.sparse.coo_array.

The input is a numpy matrix and output is a dictionary with keys equal to the tuple,
(row, column), and value is the non-zero value at that row and column.

"""

from typing import Optional, Callable, Any, Union
import numpy as np

__all__ = [
    "matrix_to_coo",
]

# Metadata
__version__ = "0.1.0"

__author__ = "Scott E. Boyce"
__email__ = "boyce@engineer.com"
__license__ = "MIT"
__status__ = "Development"  # set to "Prototype", "Development", "Production"
__maintainer__ = "Scott E. Boyce"
__url__ = "https://github.com/ScottBoyce-Python/matrix_to_coo"
__description__ = "Function that converts a numpy matrix to a coordinate list (COO) dictionary."
__copyright__ = "Copyright (c) 2024 Scott E. Boyce"


def matrix_to_coo(
    matrix: np.ndarray, one_based_index: bool = False, isclose: Optional[Callable[[Any], bool]] = None
) -> dict[tuple[int, int], Union[int, float]]:
    """
    Convert a dense matrix to a COO (Coordinate list) format.

    Args:
        matrix (np.ndarray):                       The input dense matrix to convert.
        one_based_index (bool):                    If True, use 1-based indexing for the row and
                                                   column indices; otherwise, use 0-based indexing.
        isclose (Optional[Callable[[Any], bool]]): A function to determine if a matrix value should
                                                   be considered non-zero. Defaults to checking
                                                   if the value is non-zero for integer types
                                                   or >1.0e-20 from zero for floating-point types.

    Returns:
        dict: A dictionary where keys are (row, column) tuples and
              values are the non-zero elements of the matrix.
    """
    # Define the default isclose function if none is provided
    if isclose is None:
        if matrix.dtype.kind in "ui":
            isclose = lambda x: x != 0  # Unsigned or integer non-zero values
        else:
            isclose = lambda x: x < -1.0e-20 or 1.0e-20 < x  # Float values different from zero

    it = np.nditer(matrix, flags=["multi_index"], order="F")
    if one_based_index:
        return {(it.multi_index[0] + 1, it.multi_index[1] + 1): v.item() for v in it if isclose(v)}

    return {it.multi_index: v.item() for v in it if isclose(v)}


if __name__ == "__main__":
    # Test 1: Simple integer matrix
    matrix1 = np.array(
        [
            [0, 2, 0],
            [3, 0, 4],
            [0, 0, 0],
        ],
    )
    expected1 = {
        (0, 1): 2,
        (1, 0): 3,
        (1, 2): 4,
    }

    result1 = matrix_to_coo(matrix1)

    assert result1 == expected1, f"Test 1 Failed: {result1} != {expected1}"

    # Test 2: Floating-point matrix with small values
    matrix2 = np.array(
        [
            [0.0, 2.1, 0.0],
            [3.2, 0.0, 0.0],
            [0.0, 0.0, 1e-21],
        ]
    )
    expected2 = {
        (0, 1): 2.1,
        (1, 0): 3.2,
    }

    result2 = matrix_to_coo(matrix2)
    
    assert result2 == expected2, f"Test 2 Failed: {result2} != {expected2}"

    # Test 3: One-based indexing
    matrix3 = np.array(
        [
            [0, 0, 5],
            [6, 0, 0],
            [0, 0, 0],
        ],
    )
    expected3 = {
        (1, 3): 5,
        (2, 1): 6,
    }

    result3 = matrix_to_coo(matrix3, one_based_index=True)

    assert result3 == expected3, f"Test 3 Failed: {result3} != {expected3}"

    # Test 4: Custom isclose function (e.g., consider values >= 5 as non-zero)
    matrix4 = np.array(
        [
            [4, 5, 6],
            [7, 0, 0],
            [0, 8, 9],
        ]
    )
    expected4 = {
        (0, 1): 5,
        (0, 2): 6,
        (1, 0): 7,
        (2, 1): 8,
        (2, 2): 9,
    }

    result4 = matrix_to_coo(matrix4, isclose=lambda x: x >= 5)

    assert result4 == expected4, f"Test 4 Failed: {result4} != {expected4}"

    # Test 5: All zeros matrix
    matrix5 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    expected5 = {}

    result5 = matrix_to_coo(matrix5)

    assert result5 == expected5, f"Test 5 Failed: {result5} != {expected5}"

    print("All tests passed!")
