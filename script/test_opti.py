import casadi as ca

opti = ca.Opti()

class Model:
    def __init__(self) -> None:
        self.coeffs = [0.2, 1, 0.4, 1]
        
    def fx(self, t):
        t = self.gain(t)
        def make_poly(var, cs):
            f = 0
            for i, c in enumerate(reversed(cs)):
                f += c* var**i
            return f
        return make_poly(t, self.coeffs)
    
    def gain(self, tz):
        return tz*5

x = opti.variable()
y = opti.variable()

model = Model()

z = ca.SX.sym('z', 6, 2)
z[0, 0] = 234
breakpoint()

s = ca.SX.sym('s')
expr = model.fx(s)
F = ca.Function('F', [s], [expr])
f = F(x)
opti.minimize(  f   )

opti.solver('ipopt')

sol = opti.solve()
print(sol.value(x))
