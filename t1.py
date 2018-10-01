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


def converges(A: np.array) -> bool:
    dominant = False
    for i in range(len(A)):
        s = sum(A[i] - A[i][i])
        if A[i][i] < s:
            return False
        dominant = dominant or A[i][i] > s
    return dominant



def gauss_seidel(A: np.array, B: np.array, tol: float = 1.0e-10,
                 relax: float = 1.0) -> np.array:
    Xi = np.zeros_like(B)
    X = Xi.copy()
    crit = tol + 1

    k = 0
    # 344 operações de FP por iteração do algoritmo
    # assumindo que 1-relax não é recalculado a cada iteração!
    while crit > tol:
        k += 1

        # 5 operações
        X[0] = (1-relax) * X[0] + relax * (B[0] - X[1]) / A[0, 0]

        for i in range(1, len(A) // 2):
            # 24 iterações, 6 operações cada
            X[i] = (1-relax) * X[i] + relax * (B[i] - X[i-1] - X[i+1]) / A[i, i]

        for i in range(len(A) // 2, len(A) - 1):
            # 24 iterações, 6 operações cada
            X[i] = (1-relax) * X[i] + relax * (B[i] - X[i-1] - X[i+1]) / A[i, i]
        # 5 operações
        X[-1] = (1-relax) * X[-1] + relax * (B[-1] - X[-2]) / A[-1, -1]

        crit = max(abs(x - xi) for x, xi in zip(X, Xi))
        Xi = X.copy()

    return X, k


if __name__ == '__main__':
    n = 50
    A, B = create_matrix(n)

    print('# Questão A:')

    X = lu_crout(A.copy(), B.copy())

    print('Primeira e última incógnitas:', *X[[0, -1]])
    print('Resíduo máximo:', residual(A, B, X))

    lu_ops = (4 * n ** 3 + 15 * n ** 2 - 7 * n - 6) // 6
    print('Total de operações em FP:', lu_ops)
    print()

    print('# Questão B:')

    X = gauss_trid(A.copy(), B.copy())

    print('Primeira e última incógnitas:', *X[[0, -1]])
    print('Resíduo máximo:', residual(A, B, X))

    gt_ops = 8 * n - 7
    print('Total de operações em FP:', gt_ops)
    print()

    print('# Questão C:')

    if converges(A):
        print('Sistema tem convergência garantida (diagonal dominante).')
        print('É possível usar fator de relaxação para acelerar o cálculo.')
    else:
        print('Sistema pode não ter convergência (sem diagonal dominante).')

    crit = 1.0e-4
    print('Critério de parada:', crit)

    print('Testando fatores de relaxamento (0.1 até 1.9, passos de 0.1):')

    min_iter = (100, 0)
    for f in range(1, 20):
        factor = f / 10
        _, k = gauss_seidel(A.copy(), B.copy(), crit, factor)
        print('Fator:', factor, '- iterações:', k)

        min_iter = min(min_iter, (k, factor))
    iters, factor = min_iter
    print('Fator com menos iterações:', factor)

    X, k = gauss_seidel(A.copy(), B.copy(), crit, factor)
    print('Primeira e última incógnitas:', *X[[0, -1]])
    print('Resíduo máximo:', residual(A, B, X))
    print('Número de iterações:', k)

    gs_ops = 344 * k
    print('Total de operações em FP:', gs_ops)

    X2, _ = gauss_seidel(A.copy(), B.copy(), crit ** 2, factor)
    trunc = max(abs((X - X2) / X2))
    print('Erro de truncamento máximo:', trunc)
    print()

    print('# Questão D:')
    print('Operações LU-Crout:', lu_ops)
    print('Operações Gauss otimizado:', gt_ops)
    print('Operações Gauss-Seidel:', gs_ops)
    min_ = min(lu_ops, gt_ops, gs_ops)
    if min_ == lu_ops:
        best = 'LU-Crout'
    elif min_ == gt_ops:
        best = 'Gauss otimizado'
    else:
        best = 'Gauss-Seidel'
    print('Melhor método para este sistema:', best)
