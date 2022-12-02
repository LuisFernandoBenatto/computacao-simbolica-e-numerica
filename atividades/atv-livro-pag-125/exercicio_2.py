import numpy as np
import sympy

x = sympy.Symbol('x')
sympy.init_printing()


def lagrange_polynom(points):
    p = 0
    results = []
    for j, (x_j, y_j) in enumerate(points):
        result = 1
        for i, (x_i, y_i) in enumerate(points):
            if j != i:
                result = result * ((x - x_i) / (x_j - x_i))

        results.append(result)
        p = p + (result * y_j)

    return p, results


def func_a(x):
    return np.sin(np.pi * x)


def func_b(x):
    return (x - 1) ** (1/3)


def func_c(x):
    return np.log10(3*x - 1)


def func_d(x):
    return (np.e ** (2 * x)) - x


def absolute_error(p, p_):
    return abs(p - p_)


def main():
    x0 = 1
    x1 = 1.25
    x2 = 1.6
    x_target = 1.4

    X1 = [x0, x1]
    X2 = [x0, x1, x2]

    data = [('a', func_a), ('b', func_b), ('c', func_c), ('d', func_d)]
    for letter, func in data:
        y_target = func(x_target)
        y1 = [func(value) for value in X1]
        y2 = [func(value) for value in X2]
        model1 = np.poly1d(np.polyfit(X1, y1, 1))
        model2 = np.poly1d(np.polyfit(X2, y2, 2))

        y1_model = model1(x_target)
        y2_model = model2(x_target)
        error1 = absolute_error(y_target, y1_model)
        error2 = absolute_error(y_target, y2_model)

        print(f'===== {letter} | np.poly1d =====')
        print('Modelo Linear: ')
        print(model1)
        print(f'y1 modelo linear: {y1_model}')
        print(f'modelo linear erro: {error1}')
        print()
        print('Modelo Grau 2: ')
        print(model2)
        print(f'Y2: {y2_model}')
        print(f'Modelo Grau 2 Erro: {error2}')

        p1_lagrange, result_1 = lagrange_polynom(
                [(x_, y_) for x_, y_ in zip(X1, y1)])
        p1_lagrange_expanded = sympy.expand(p1_lagrange)
        p2_lagrange, result_2 = lagrange_polynom(
                [(x_, y_) for x_, y_ in zip(X2, y2)])
        p2_lagrange_expanded = sympy.expand(p2_lagrange)

        y1_lagrange = p1_lagrange_expanded.subs({x: x_target}).n()
        y2_lagrange = p2_lagrange_expanded.subs({x: x_target}).n()
        error1_lagrange = absolute_error(y_target, y1_lagrange)
        error2_lagrange = absolute_error(y_target, y2_lagrange)

        print()
        print(f'===== {letter} - Lagrange =====')
        print('Modelo Linear: ')
        print(p1_lagrange_expanded)
        print(f'y1 modelo linear: {y1_lagrange}')
        print(f'modelo linear erro: {error1_lagrange}')
        print()
        print('Modelo Grau 2: ')
        print(sympy.expand(p2_lagrange))
        print(f'Y2: {y2_lagrange}')
        print(f'Modelo Grau 2 Erro: {error2_lagrange}')
        print()


if __name__ == '__main__':
    main()
