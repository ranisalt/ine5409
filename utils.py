def func_poly(vec, x):
    y = vec[0]
    for i in vec[1:]:
        y = y * x + i
    return y

def residue(func, root):
    return abs(func(root))
