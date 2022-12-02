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
    xa_target = 8.4
    points_a = [(8.1, 16.94410), (8.3, 17.56492),
                (8.6, 18.50515), (8.7, 18.82091)]
    # b)
    xb_target = - 1 / 3
    points_b = [(-0.75, -0.07181250), (-0.5, -0.02475000),
                (-0.25, 0.33493750), (0, 1.10100000)]
    # c)
    xc_target = 0.25
    points_c = [(0.1, 0.62049958), (0.2, -0.28398668),
                (0.3, 0.00660095), (0.4, 0.24842440)]
    # d)
    xd_target = 0.9
    points_d = [(0.6, -0.17694460), (0.7, 0.01375227),
                (0.8, 0.22363362), (1.0, 0.65809197)]

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
