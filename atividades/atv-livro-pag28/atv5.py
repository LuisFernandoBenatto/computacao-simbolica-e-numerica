import math

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


def get_data():
    a = 4/5 + 1/3
    b = 4/5 * 1/3
    c = (1/3 - 3/11) + 3/20
    d = (1/3 + 3/11) - 3/20
    data = [('a', a, truncate(a, 3), round(a, 3)),
            ('b', b, truncate(b, 3), round(b, 3)),
            ('c', c, truncate(c, 3), round(c, 3)),
            ('d', d, truncate(d, 3), round(d, 3))]
    return data


def main():
    data = get_data()
    print(f"{data=}\n")

    for item in data:
        letter, p, p_truncate, p_round = item

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
        print()


if __name__ == '__main__':
    main()
