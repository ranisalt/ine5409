import functools
import operator
import numpy as np


def create_matrix(n: int) -> [np.array, np.array]:
    '''
    para i=1;         x(i)+x(i+1)=1.50
    para i=2:n/2      x(i-1)+4x(i)+x(i+1)=1.00
    para i=n/2+1:n-1  x(i-1)+5x(i)+x(i+1)=2.00
    para i=n          x(i-1)+x(i)=3.00
    '''
    A, B = np.zeros((n, n), dtype=float), np.zeros(n, dtype=float)

    A[0, 0], A[0, 1], B[0] = 1, 1, 1.5
    for i in range(1, n//2):
        A[i, i-1], A[i, i], A[i, i+1], B[i] = 1, 4, 1, 1.0
    for i in range(n//2, n-1):
        A[i, i-1], A[i, i], A[i, i+1], B[i] = 1, 5, 1, 2.0
    A[-1, -2], A[-1, -1], B[-1] = 1, 1, 3.0

    return A, B


def optimize(A: np.array, B: np.array) -> np.array:
    '''
    >>> A = np.array([[1, -1,  0, 0], \
                      [1,  1, -1, 0], \
                      [0,  1, -1, 1], \
                      [0,  0, -1, 1]], dtype=float)
    >>> B = np.array([0, 1, 2, -1], dtype=float)
    >>> optimize(A, B)
    (array([nan,  1.,  1., -1.]), array([ 1.,  1., -1.,  1.]), array([-1., -1.,  1., nan]), array([ 0.,  1.,  2., -1.]))
    '''
    t = np.append([np.nan], A.diagonal(-1))
    r = A.diagonal(0).copy()
    d = np.append(A.diagonal(1), [np.nan])
    return t, r, d, B


def residual(A: np.array, B: np.array, X: np.array) -> float:
    return max(abs(sum(A[i, j] * X[j] for j in range(len(X))) - B[i])
               for i in range(len(B)))


def lu_crout(A: np.array, B: np.array) -> np.array:
    '''
    >>> A = np.array([[0, 1, 2], [2, -1, -1], [1, 0, 1]], dtype=float)
    >>> B = np.array([1, 2, 3])
    >>> lu_crout(A, B)
    array([ 0, -5,  3])
    '''
    n = len(A)

    # A, B = lu_decompose(A, B)
    for k in range(n):
        for i in range(k, n):
            A[i, k] = A[i, k] - sum(A[i, r] * A[r, k] for r in range(k))

        # A, B = pivot(A, B, k)
        vmax, imax = max((abs(A[i, k]), i) for i in range(k, n))

        A[[k, imax]] = A[[imax, k]]
        B[[k, imax]] = B[[imax, k]]

        for j in range(k+1, n):
            A[k, j] = (A[k, j] - sum(A[k, r] * A[r, j] for r in range(k))) / A[k, k]

    # X = lu_solve(A, B)
    C = np.empty_like(B)
    C[0] = B[0] / A[0, 0]
    for i in range(1, n):
        C[i] = (B[i] - sum(A[i, :i] * C[:i])) / A[i, i]

    X = np.empty_like(B)
    X[-1] = C[-1]
    for i in reversed(range(0, n-1)):
        X[i] = C[i] - sum(A[i, i+1:] * X[i+1:])

    return X


def gauss_trid(A: np.array, B: np.array) -> np.array:
    '''
    >>> A = np.array([[1, -1,  0,  0,  0], \
                      [1,  1, -1,  0,  0], \
                      [0,  1, -1,  1,  0], \
                      [0,  0, -1,  1,  1], \
                      [0,  0,  0, -1,  2]], dtype=float)
    >>> B = np.array([0, 1, 2, -1, -2], dtype=float)
    >>> gauss_trid(A, B)
    array([5., 5., 9., 6., 2.])
    '''
    n = len(A)

    # t, r, d, b = optimize(A, B)
    t = np.append([np.nan], A.diagonal(-1))
    r = A.diagonal(0).copy()
    d = np.append(A.diagonal(1), [np.nan])
    for i in range(1, n):
        factor = t[i] / r[i-1]
        r[i] -= factor * d[i-1]
        B[i] -= factor * B[i-1]

    X = np.empty_like(B)
    X[-1] = B[-1] / r[-1]
    for i in reversed(range(n-1)):
        X[i] = (B[i] - d[i] * X[i+1]) / r[i]

    return X


def gauss_seidel(A: np.array, B: np.array, Xi: np.array,
                 tolerance: float = 1e-10, relax: float = 0.8) -> np.array:
    '''
    # >>> A = np.array([[1, 0, 1, 0, 0,   0], \
    #                   [1, 2, 0, 1, 0,   0], \
    #                   [0, 1, 2, 0, 1,   0], \
    #                   [0, 1, 0, 2, 1,   0], \
    #                   [0, 0, 1, 0, 2,   1], \
    #                   [0, 0, 0, 1, 0, 0.5]], dtype=float)
    # >>> B = np.array([4, 2, 2, 3, 3, 1], dtype=float)
    # >>> Xi = np.zeros_like(B)
    # >>> gauss_seidel(A, B, Xi, 1e-5)
    # array([3.1, -1.24, 9., 1.4, 1.45, -8.])
    '''
    X = Xi.copy()
    crit = tolerance + 1

    k = 0
    while crit > tolerance and k < 100:
        k += 1

        X[0] = (1-relax) * X[0] + relax * (B[0] - X[1]) / A[0, 0]

        for i in range(1, len(A) // 2):
            X[i] = (1-relax) * X[i] + relax * (B[i] - X[i-1] - X[i+1]) / A[i, i]

        for i in range(len(A) // 2, len(A) - 1):
            X[i] = (1-relax) * X[i] + relax * (B[i] - X[i-1] - X[i+1]) / A[i, i]
        X[-1] = (1-relax) * X[-1] + relax * (B[-1] - X[-2]) / A[-1, -1]

        crit = max(abs(x - xi) for x, xi in zip(X, Xi))
        Xi = X.copy()

    print(crit, k)

    return X


if __name__ == '__main__':
    n = 50
    A, B = create_matrix(n)

    print('# Questão A:')

    X = lu_crout(A.copy(), B.copy())

    print('Primeira e última incógnitas: {} {}'.format(*X[[0, -1]]))
    print('Resíduo máximo: {}'.format(residual(A, B, X)))
    print('Total de operações: {}'.format((4*n**3 + 15*n**2 - 7*n - 6) // 6))
    print()

    print('# Questão B:')

    X = gauss_trid(A.copy(), B.copy())

    print('Primeira e última incógnitas: {} {}'.format(*X[[0, -1]]))
    print('Resíduo máximo: {}'.format(residual(A, B, X)))
    print('Total de operações: {}'.format(8*n - 7))
    print()
