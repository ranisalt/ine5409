import math
from random import uniform

def func_poly(vec):
    '''
    >>> f = func_poly([1, -3, 3, -1])
    >>> f(0), f(1), f(2)
    (-1, 0, 1)
    '''
    def wrapped(x):
        it = iter(vec)
        y = next(it) * x + next(it)
        for i in it:
            y = y * x + i
        return y
    return wrapped

def find_roots_poly_singular(n, vec):
    '''
    >>> roots = [*find_roots_poly_singular(3, [1, -3, 3, -1])]
    >>> roots[0]
    (1.0133807073946937+0j)
    >>> len(roots)
    3
    '''
    r = 1 + max(abs(c) for c in vec[1:n + 2]) / vec[0]
    func = func_poly(vec)

    start, stop, step, k = -r, r, 1 / (10 * math.pi), 0
    while start < stop:
        i, j = start, start + step
        a, b = func(i), func(j)
        if a * b <= 0:
            k += 1
            yield (i + j) / 2 + 0j
        start = j

    for i in range(k, n):
        yield complex(uniform(-0.5, 0.5) / 2 * r, uniform(-0.5, 0.5) / 2 * r)

def remainder(n, vec, x, n_div=None):
    '''
    >>> roots = [*find_roots_poly_singular(3, [1, -3, 3, -1])]
    >>> next(remainder(3, [1, -3, 3, -1], roots[0], 1))
    (2.3957264150276103e-06+0j)
    >>> [*remainder(3, [1, -3, 3, -1], roots[0])]
    [(2.3957264150276103e-06+0j), (0.0005371299911474114+0j), (0.040142122184080975+0j), 1]
    '''
    if not n_div:
        n_div = n + 1

    k = 0

    while n > 0 and k < n_div:
        k += 1

        b = [vec[0]]
        for i in vec[1:]:
            b.append(i + x * b[-1])

        yield b[n]

        n -= 1
        vec = b

    yield vec[0]

def newton_poly_singular(n, vec, x, tolerance = 1e-14):
    '''
    >>> roots = [*find_roots_poly_singular(3, [1, -3, 3, -1])]
    >>> newton_poly_singular(3, [1, -3, 3, -1], roots[0])
    (1.0000024185919845+0j)
    '''
    dx = 2 * tolerance

    k = 0
    while abs(dx) > tolerance and k < 50:
        k += 1
        r0, r1, *_ = remainder(n, vec, x)
        dx = -r0 / r1
        x += dx
    return x

def roots(vec):
    n = len(vec) - 1
    vec = [v / vec[0] for v in vec]
    xi = [*find_roots_poly_singular(n, vec)]

    for x in xi:
        yield newton_poly_singular(n, vec, x)
