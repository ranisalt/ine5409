import math
import operator

import numeric, poly_singular, utils

def f1(x):
    return x ** 3 - math.exp(x)

# def f1_2nd_derivative(x):
#     return 6 * x - math.exp(x)

if __name__ == '__main__':
    print('Questão 1')

    my_roots = numeric.roots(f1, -10, 10)

    for root in my_roots:
        print('raiz =', root, '/ resíduo =', utils.residue(f1, root))

    print()

    print('Questão 2')
    vec = [1, -7, 20.95, -34.75, 34.5004, -20.5012, 6.7512, -0.9504, 0, 0, 0]

    my_roots = poly_singular.roots(vec)
    for root in sorted(my_roots, key=operator.attrgetter('real'), reverse=True):
        print('raiz =', root, '/ resíduo =',
              utils.residue(utils.func_poly(vec), root))
