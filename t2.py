import math
import operator

import numeric, poly, utils

def f1(x):
    return x ** 3 - math.exp(x)

def f1_2nd_derivative(x):
    return 6 * x - math.exp(x)

if __name__ == '__main__':
    print('Questão 1')

    my_roots = numeric.roots(f1, -10, 10)
    for root in my_roots:
        print('raiz =', root, '/ resíduo =', utils.residue(f1, root))

    print('Pontos de descontinuidade:')
    my_discontinuities = numeric.roots(f1_2nd_derivative, -10, 10)
    for root in my_discontinuities:
        print('x =', root)

    print()
    vec = [1, -7, 20.95, -34.75, 34.5004, -20.5012, 6.7512, -0.9504, 0, 0, 0]
    # vec = [1, -3, 3, 1]

    print('Questão 2a')

    my_roots = poly.roots(vec)
    for root, _ in zip(*my_roots):
        print('raiz =', root, '/ resíduo =',
              abs(utils.func_poly(vec, root)))

    print()

    print('Questão 2b')

    my_roots = poly.roots(vec, singular=False)
    factors = []
    for root, m in zip(*my_roots):
        print('raiz =', root, '/ M =', m, '/ resíduo =',
              abs(utils.func_poly(vec, root)))

        for _ in range(m):
            factors.append(round(root * 10) / 10)

    print()

    print('Questão 2c')
    print(''.join(['(x-{})'.format(f) for f in factors]))

    print('Questão 2d')
    xOctave = [
        '-9.600641510593409e+00 + 0.000000000000000e+00i',
        '3.740640443692235e-01 + 1.019877558680444e+00i',
        '3.740640443692235e-01 - 1.019877558680444e+00i',
        '4.582693811675116e-01 + 3.907275181778783e-01i',
        '4.582693811675116e-01 - 3.907275181778783e-01i',
        '4.679873297599668e-01 + 1.108483429040146e-01i',
        '4.679873297599668e-01 - 1.108483429040146e-01i',
        '0.000000000000000e+00 + 0.000000000000000e+00i',
        '0.000000000000000e+00 + 0.000000000000000e+00i',
        '0.000000000000000e+00 + 0.000000000000000e+00i',
    ]
    for root in xOctave:
        print(root)
    print('Obs: obtidas usando roots([', *vec, '])')
