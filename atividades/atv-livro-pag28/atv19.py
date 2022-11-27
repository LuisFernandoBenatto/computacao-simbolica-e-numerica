from sympy import symbols


def main():
    s = symbols('s')
    c = symbols('c')
    f = symbols('f')
    signal_expr = (-1) ** s
    number_expr = (2 ** (c - 1023)) * (1 + f)
    expr = signal_expr * number_expr

    data = [
        ('a)', '0', '10000001010', '1001001100000000000000000000000000000000000000000000'),
        ('b)', '1', '10000001010', '1001001100000000000000000000000000000000000000000000'),
        ('c)', '0', '01111111111', '0101001100000000000000000000000000000000000000000000'),
        ('d)', '0', '01111111111', '0101001100000000000000000000000000000000000000000001'),
        ]

    for item in data:
        letter, signal, integer_part, float_part = item
        signal = int(signal)

        integer_result = 0
        for index, number in enumerate(integer_part):
            exp = 10 - index
            number = int(number)
            result = number * (2 ** exp)
            integer_result += result

        float_result = 0
        for index, number in enumerate(float_part):
            exp = index + 1
            number = int(number)
            result = number * (1/2 ** exp)
            float_result += result

        expr_result = expr.subs(
                {s: signal, c: integer_result, f: float_result}).n()
        print(f"{letter} {expr_result=} "
              f"| {signal=} {integer_result=} {float_result=}")


if __name__ == '__main__':
    main()
