def func_poly(vec):
    def wrapped(x):
        it = iter(vec)
        y = next(it) * x + next(it)
        for i in it:
            y = y * x + i
        return y
    return wrapped

def residue(func, root):
    return abs(func(root))
