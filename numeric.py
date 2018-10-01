import math
from random import uniform

def find_roots(func, start, stop, step, limit):
    while start < stop:
        i, j = start, start + step
        a, b = func(i), func(j)
        if a * b < 0 and a < limit and b < limit:
            yield (i + j) / 2
        start = j

def newton_numeric(func, xi, tolerance = 1e-14):
    dx = 2 * tolerance

    k = 0
    while abs(dx) > tolerance and k < 50:
        k += 1
        val = func(xi)
        dx = -val * dx / (func(xi + dx) - val)
        xi += dx

    return xi

def roots(func, start, stop, step = 0.01, limit = 0.5):
    unoptimized = find_roots(func, start, stop, step, limit)

    for root in unoptimized:
        yield newton_numeric(func, root)
