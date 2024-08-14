# matrix_to_coo

This module provides a function to convert a dense numpy matrix to a Coordinate list (COO) format. The COO format stores a list of (row, column, value) for non-zero values in a matrix, similar to `scipy.sparse.coo_array`. 

The input is a numpy matrix and output is a dictionary with keys equal to the tuple, `(row, column)`, and `value` is the non-zero value at that row and column.



## Installation
To install the module
```bash
pip install --upgrade git+https://github.com/ScottBoyce-Python/matrix_to_coo.git
```

or you can clone the respository with
```bash
git clone https://github.com/ScottBoyce-Python/matrix_to_coo.git
```
and then move the file `matrix_to_coo/matrix_to_coo.py` to wherever you want to use it.


## Usage


```python
from matrix_to_coo import matrix_to_coo
import numpy as np

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

assert result1 == expected1

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

assert result2 == expected2

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

assert result3 == expected3

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

assert result4 == expected4

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

assert result5 == expected5



```

&nbsp; 

## Testing

This project just has several simple tests in the `if __name__ == "__main__"` portion of the `__init__.py` file. To run the tests type:  
`python matrix_to_coo/__init__.py`   

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Author
Scott E. Boyce
