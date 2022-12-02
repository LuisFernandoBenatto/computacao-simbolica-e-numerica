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


def main():
    # a)
    xa_target = 0.43
    points_a = [(0, 1), (0.25, 1.64872), (0.5, 2.71828), (0.75, 4.48169)]
    # b)
    xb_target = 0
    points_b = [(-0.5, 1.93750), (-0.25, 1.332303),
                (0.25, 0.800781), (0.5, 0.687500)]
    # c)
    xc_target = 0.18
    points_c = [(0.1, -0.29004986), (0.2, -0.56079734),
                (0.3, -0.81401972), (0.4, -1.0526302)]
    # d)
    xd_target = 0.25
    points_d = [(-1, 0.86199480), (-0.5, 0.95802009),
                (0, 1.0986123), (0.5, 1.2943767)]

    data = [('a', xa_target, points_a), ('b', xb_target, points_b),
            ('c', xc_target, points_c), ('d', xd_target, points_d)]
    for letter, x_target, points in data:
        p1_lagrange, result_1 = lagrange_polynom(points[:2])
        p1_lagrange_expanded = sympy.expand(p1_lagrange)

        p2_lagrange, result_2 = lagrange_polynom(points[:3])
        p2_lagrange_expanded = sympy.expand(p2_lagrange)

        p3_lagrange, result_2 = lagrange_polynom(points)
        p3_lagrange_expanded = sympy.expand(p3_lagrange)

        y1_lagrange = p1_lagrange_expanded.subs({x: x_target}).n()
        y2_lagrange = p2_lagrange_expanded.subs({x: x_target}).n()
        y3_lagrange = p3_lagrange_expanded.subs({x: x_target}).n()

        print(f'===== {letter} - Lagrange =====')
        print('Modelo Linear: ')
        print(p1_lagrange_expanded)
        print(f'Y1: {y1_lagrange}')
        print()
        print('Modelo Grau 2: ')
        print(sympy.expand(p2_lagrange))
        print(f'Y2: {y2_lagrange}')
        print()
        print('Modelo Grau 3: ')
        print(sympy.expand(p3_lagrange))
        print(f'Y2: {y3_lagrange}')
        print()


if __name__ == '__main__':
    main()
