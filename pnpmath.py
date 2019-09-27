# -*- coding: utf-8 -*-
"""
pyNumbaPacking compiled math library
"""
import numpy as np
import numba


@numba.jit(numba.float64(numba.int32), nopython=True)
def half_integer_gamma(x):
    """
    halfIntegerGamma(x) = gamma(x / 2)
    x must be an integer
    """
    result = np.sqrt(np.pi) if x % 2 else 1
    for i in range(2 - x % 2, x, 2):
        result *= i / 2
    return result


@numba.jit(numba.float64(numba.int32), nopython=True)
def sphere_volume_constant(d):
    return np.pi ** (d / 2) / half_integer_gamma(d + 2)


@numba.jit(
    numba.float64(
        numba.float64[:], numba.boolean[:], numba.boolean[:], numba.int32
    ),
    nopython=True
)
def cofactor(A, mask1, mask2, n):
    """
    determines the partial determinant based on a series of index masks

    args:
        A (float64[:], n**2): square matrix to evaluate, flattened
        mask1, mask2 (bool[:], n): masks for included rows/columns
        n (int32): size of matrix.
    """
    # Finds the first unmasked index
    for i in range(n):
        if mask1[i]:
            break
    else:
        return 1
    mask1[i] = False
    total = 0
    sign = 1
    for j in range(n):
        if mask2[j]:
            mask2[j] = False
            total += sign * A[i * n + j] * cofactor(A, mask1, mask2, n)
            mask2[j] = True
            sign = -sign
    mask1[i] = True
    return total


@numba.jit(
    numba.void(numba.float64[:], numba.float64[:], numba.int32),
    nopython=True
)
def adjugate_transpose(A, A_adjt, n):
    """
    Computes the adjugate transpose (elementwise derivative of determinant)
    of a matrix

    args:
        A (float64[:], n**2): square matrix, flattened
        A_inv (float64[:], n**2): output matrix, flattened
        n (int32): size of matrices.
    """
    mask1 = np.ones(n, dtype=np.bool8)
    mask2 = np.ones(n, dtype=np.bool8)
    for i in range(n):
        for j in range(n):
            mask1[i] = False
            mask2[j] = False
            A_adjt[i*n+j] = (
                (1 - 2 * ((i + j) % 2)) *
                cofactor(A, mask1, mask2, n)
            )
            mask1[i] = True
            mask2[j] = True


@numba.jit(numba.float64(numba.float64[:], numba.int32), nopython=True)
def determinant(A, n):
    """
    computes the determinant of a flattened n x n matrix A

    args:
        A (float64[:], n**2): square matrix to evaluate, flattened
        n (int32): size of matrix.
    """
    mask1 = np.ones(n, dtype=np.bool8)
    mask2 = np.ones(n, dtype=np.bool8)
    return cofactor(A, mask1, mask2, n)


@numba.jit(
    numba.void(numba.float64[:], numba.float64[:], numba.int32),
    nopython=True
)
def inverse(A, A_inv, n):
    """
    computes the inverse of a flattened n x n matrix A

    args:
        A (float64[:], n**2): square matrix to invert, flattened
        A_inv (float64[:], n**2): output matrix, flattened
        n (int32): size of matrices.
    """
    mask1 = np.ones(n, dtype=np.bool8)
    mask2 = np.ones(n, dtype=np.bool8)
    sign = 1
    det = determinant(A, n)
    for i in range(n):
        for j in range(n):
            mask1[i] = False
            mask2[j] = False
            A_inv[j * n + i] = sign * cofactor(A, mask1, mask2, n) / det
            mask1[i] = True
            mask2[j] = True
            sign = -sign


@numba.jit(
    numba.float64(numba.float64[:, :], numba.float64[:], numba.int32),
    nopython=True
)
def polyhedron_volume(A, b, good_rows):
    """
    Recursively computes the volume of a convex polyhedron about the origin
    defined bt A @ x <= b

    Ref: J. B. Lasserre, "An Analytical Expression and an Algorithm
    for the Volume of a Convex Polyhedron in Rn", 1983

    params:
        A (np.float64[m,n]): matrix of contraint covectors
        b (np.float64[m]: vector of constraint offsets
        good_rows (np.int32): restricts the contraints under consideration
    :return:
    """
    # recursion terminates on the 1d case, which can be solved explicitly
    volume = 0
    if A.shape[1] == 1:
        # Each constraint defines a lower or upper bound
        lower, upper = -np.inf, np.inf
        for i in range(good_rows):
            if A[i, 0] > 0:
                upper = np.minimum(upper, b[i] / A[i, 0])
            elif A[i, 0] < 0:
                lower = np.maximum(lower, b[i] / A[i, 0])
        volume = np.maximum(0, upper - lower)
    # for dim > 1, recur
    else:
        # allocate arrays for sub_problems
        new_A = np.empty((good_rows-1, A.shape[1]-1), dtype=np.float64)
        new_b = np.empty(good_rows-1, dtype=np.float64)
        # loop through constraints
        i = 0
        while i < good_rows:
            # find suitable direction to project out
            j = np.argmax(np.abs(A[i, :]))
            # skip indices that are numerically unstable
            if np.abs(A[i, j]) < 1e-10:
                good_rows -= 1
                A[i, :], A[good_rows, :] = A[good_rows, :], A[i, :]
                b[i], b[good_rows] = b[good_rows], b[i]
                continue
            # Construct the projected subproblem
            for k in range(good_rows):
                if k == i:
                    continue
                new_b[k - (k > i)] = b[k] - A[k, j] * b[i] / A[i, j]
                for l in range(A.shape[1]):
                    if l == j:
                        continue
                    new_A[k - (k > i), l - (l > j)] = (
                        A[k, l] - A[k, j] * A[i, l] / A[i, j]
                    )
            # compute area without prefeactors
            area = polyhedron_volume(new_A, new_b, good_rows-1)
            # if area is zero, constraint can safely be removed
            if area == 0:
                good_rows -= 1
                A[i, :], A[good_rows, :] = A[good_rows, :], A[i, :]
                b[i], b[good_rows] = b[good_rows], b[i]
            else:
                volume += area * b[i] / (np.abs(A[i, j]) * A.shape[1])
                i += 1
    return volume
