from sympy import factorial, Symbol, Sum
import numpy as np


def real_error(p, p_):
    return p - p_


def absolute_error(p, p_):
    return abs(p - p_)


def relative_error(p, p_):
    return abs(p - p_) / abs(p)


def main():
    n = Symbol('n', positive=True)
    exp_expr = 1 / factorial(n)
    sum_exp_a = Sum(exp_expr, (n, 0, 5))
    sum_exp_b = Sum(exp_expr, (n, 0, 10))

    data = [('a', np.e, sum_exp_a.n()),
            ('b', np.e, sum_exp_b.n())]
    for item in data:
        letter, p, p_ = item
        real_error_ = real_error(p, p_)
        abs_error = absolute_error(p, p_)
        relative_error_ = relative_error(p, p_)
        print(f"{letter} | erro real = {real_error_} "
              f"| erro absoluto = {abs_error} "
              f"| erro relativo = {relative_error_}")


if __name__ == '__main__':
    main()
