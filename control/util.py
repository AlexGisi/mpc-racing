import casadi as ca


def make_poly(variable, coeffs):
    f = 0
    for i, c in enumerate(reversed(coeffs)):  # coeffs are highest order to lowest
        f += c* variable**i
    return f

def deg2rad(z):
    return (z / 360) * 2 * 3.14
