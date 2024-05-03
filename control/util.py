import casadi as ca


def make_poly(variable, coeffs):
    f = 0
    for i, c in enumerate(reversed(coeffs)):  # coeffs are highest order to lowest
        f += c* variable**i
    return f

def deg2rad(z):
    return (z / 360) * 2 * 3.14

def normal_pdf(x, mean, variance):
    coeff = 1 / ca.sqrt(2 * ca.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coeff * ca.exp(exponent)
