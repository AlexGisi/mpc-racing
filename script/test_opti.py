import casadi as ca

opti = ca.Opti()

x = opti.variable()
y = opti.variable()

coeffs = [0.2, 1, 0.4, 1]
coeffs2 = [1, 0.6, 1]
def make_poly(var, cs):
    f = 0
    for i, c in enumerate(reversed(cs)):
        f += c* var**i
    return f

s = ca.SX.sym('s')
F = ca.Function('F', [s], [make_poly(s, coeffs)])
F2 = ca.Function('F', [s], [make_poly(s, coeffs2)])
f = F(x)
f1 = F2(x)
opti.minimize(  ca.gradient(f, x) + f1   )

opti.solver('ipopt')

sol = opti.solve()
print(sol.value(x))
