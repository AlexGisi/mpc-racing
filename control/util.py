def make_poly(variable, coeffs):
    f = 0
    for i, c in enumerate(reversed(coeffs)):  # coeffs are highest order to lowest
        f += c* variable**i
    return f