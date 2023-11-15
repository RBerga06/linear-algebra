# `rberga06-linear-algebra`
A Python library that tries to implement various linear algebra concepts
(inspired by a genious friend's idea)

> [!WARNING]
> This project is just a proof-of-concept and shouldn't be used in production code.
> Also, it may get discontinued and eventually archived at any point in time.
>

## Installation

`pip install 'rberga06-linear-algebra @ git+https://github.com/rberga06/linear-algebra'`

## Usage / Examples

```python
# Import everything from the library
from rberga06.linalg import *

# Play around with vectors
v1 = Vec([1, 2, 3])   # [1 2 3]
v2 = Vec.fill(1j, 5)  # [1j 1j 1j 1j 1j]

# Play around with matrices
m1 = Mat([[1,2,3],[4,5,6]])
m1[0,:]  # select the first row
m1[:,0]  # select the first column

# select the first (0) and third (2) elements
#   of the even (::2) rows:
m1[::2,(0,2)]
#   or, equivalently (since m[::2,:] is another Mat):
m1[::2,:][0,2]
```

For more examples, please have a look at the tests.
