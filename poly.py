import math
from random import uniform

from utils import func_poly

def find_roots_poly(n, vec):
    '''
    >>> roots = [*find_roots_poly(3, [1, -3, 3, -1])]
    >>> roots[0]
    (1.0133807073946937+0j)
    >>> len(roots)
    3
    '''
    xi = []
    r = 1 + max(map(abs, vec[1:])) / vec[0]

    start, stop, step = -r, r, 1 / (10 * math.pi)
    while start < stop:
        i, j = start, start + step
        a, b = func_poly(vec, i), func_poly(vec, j)
        if a * b <= 0:
            xi.append((i + j) / 2)
        start = j

    for i in range(len(xi) + 1, len(vec)):
        xi.append(uniform(-0.5, 0.5) / 2 * r)
    return xi

def briot_ruffini(n, vec, x):
    '''
    >>> briot_ruffini(3, [2, 3, 0, -4], -1)
    (2, [2, 1, -1, -3])
    >>> briot_ruffini(3, [1, -3, 3, -1], 1)
    (2, [1, -2, 1, 0])
    '''
    b = [0.0] * n
    for i in range(n):
        b[i] = vec[i] + x * b[i - 1]
    return n - 1, b

def multiplicity(r, limit=1e-7):
    '''
    >>> multiplicity([1e-10, 1e-6, 1e-4, 1])
    2
    >>> multiplicity([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1])
    5
    '''
    sum, m = abs(r[0]) + abs(r[1]), 1
    while sum < limit:
        m += 1
        sum += abs(r[m])
    return m

def remainder(vec, x):
    '''
    >>> roots = [*find_roots_poly(3, [1, -3, 3, -1])]
    >>> remainder([1, -3, 3, -1], roots[0])
    [(2.3957264150276103e-06+0j), (0.0005371299911474114+0j), (0.040142122184080975+0j), 1]
    '''
    n = len(vec) - 1
    n_div = len(vec)
    r = [0] * n_div
    r[-1] = vec[0]

    k = 0
    while n > 0 and k < n_div:
        b = [vec[0]]
        for i in range(1, n + 1):
            b.append(vec[i] + x * b[i - 1])

        r[k] = b[n]

        vec = b
        k += 1
        n -= 1
    return r

def newton_poly(vec, x, tolerance = 1e-14, singular=True):
    '''
    >>> roots = [*find_roots_poly(3, [1, -3, 3, -1])]
    >>> newton_poly(3, [1, -3, 3, -1], roots[0])
    ((1.0000024185919845+0j), 1)
    '''
    dx = 2 * tolerance

    m, k = 1, 0
    while abs(dx) > tolerance and k < 50:
        k += 1
        r = remainder(vec, x)
        m = 1 if singular else multiplicity(r)
        dx = -r[m - 1] / (m * r[m])
        x += dx
    return x, m

def roots(vec, singular=True):
    n = len(vec) - 1
    vec = [v / vec[0] for v in vec]
    xi = find_roots_poly(n, vec)

    x = [0.0] * n
    m = [0] * n

    k = 0
    while n > 0:
        x[k], m[k] = newton_poly(vec, xi[k], singular=singular)

        for _ in range(m[k]):
            n, vec = briot_ruffini(n, vec, x[k])
        k += 1

    return x[:k], m[:k]
