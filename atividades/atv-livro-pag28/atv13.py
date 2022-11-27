import math

from sympy import limit, sin, cos, symbols, oo


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
    f_x = (x * cos(x) - sin(x)) / (x - sin(x))
    limit_to_infinity = limit(f_x, x, oo).n()
    limit_to_0 = limit(f_x, x, 0).n()
    limit_to_01 = limit(f_x, x, 0.1).n()
    print(f'Expr = {f_x}')
    print(f'Limit to infinity = {limit_to_infinity}')
    print(f'Limit to 0 = {limit_to_0}')
    print(f'Limit to 1 = {limit_to_01}')

    letter, p, p_truncate, p_round = (
            'b', limit_to_01, round(limit_to_01, 4), truncate(limit_to_01, 4))

    real_error_truncate = real_error(p, p_truncate)
    real_error_round = real_error(p, p_round)

    abs_error_truncate = absolute_error(p, p_truncate)
    abs_error_round = absolute_error(p, p_round)

    relative_error_truncate = relative_error(p, p_truncate)
    relative_error_round = relative_error(p, p_round)

    print(f"{letter} "
          f"| {real_error_truncate=} "
          f"| {real_error_round=} ")
    print(f"  | {abs_error_truncate=} "
          f"  | {abs_error_round=} ")
    print(f"  | {relative_error_truncate=} "
          f"  | {relative_error_round=} ")


if __name__ == '__main__':
    main()
