import math

from sympy import factorial, limit, symbols, oo, Sum
import numpy as np


def real_error(p, p_):
    return p - p_


def absolute_error(p, p_):
    return abs(p - p_)


def relative_error(p, p_):
    return abs(p - p_) / abs(p)


def truncate(number, digits) -> float:
    nb_decimals = len(str(number).split('.')[1])
    if nb_decimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def main():
    x = symbols('x')
    e1 = np.e ** x
    e2 = np.e ** -x
    f_x = (e1 - e2) / x
    limit_to_infinity = limit(f_x, x, oo).n()
    limit_to_0 = limit(f_x, x, 0).n()
    print(f'Expr = {f_x}')
    print(f'Limit to infinity = {limit_to_infinity}')
    print(f'a) Limit to 0 = {limit_to_0}')

    n = symbols('n')
    maclaurin_expr = x ** n / factorial(n)
    maclaurin_expr_negative = (-x) ** n / factorial(n)
    sum_maclaurin = Sum(maclaurin_expr, (n, 0, 3))
    sum_maclaurin_negative = Sum(maclaurin_expr_negative, (n, 0, 3))

    diff = sum_maclaurin - sum_maclaurin_negative
    f_x_in_maclaurin = diff / x

    f_x_result = f_x.subs({x: 0.1}).n()
    f_x_in_maclaurin_result = f_x_in_maclaurin.subs({x: 0.1}).n()
    print(f'b) {f_x_result}')
    print(f'c) {f_x_in_maclaurin_result}')

    real_value = 2.003335000
    letter, p, p_, p_maclaurin = (
            'd', real_value, f_x_result, f_x_in_maclaurin_result)

    real_error_f_x = real_error(p, p_)
    real_error_f_x_maclaurin = real_error(p, p_maclaurin)

    abs_error_f_x = absolute_error(p, p_)
    abs_error_f_x_maclaurin = absolute_error(p, p_maclaurin)

    relative_error_f_x = relative_error(p, p_)
    relative_error_f_x_maclaurin = relative_error(p, p_maclaurin)

    print(f"{letter} "
          f"| {real_error_f_x=} "
          f"| {real_error_f_x_maclaurin=} ")
    print(f"  | {abs_error_f_x=} "
          f"  | {abs_error_f_x_maclaurin=} ")
    print(f"  | {relative_error_f_x=} "
          f"  | {relative_error_f_x_maclaurin=} ")


if __name__ == '__main__':
    main()
