import qutip as qt
import numpy as np

def commutator(A, B):
	return A*B - B*A
	
def anticommutator(A, B):
	return A*B + B*A

c = qt.destroy(2)
print(anticommutator(c, c.dag()) == qt.identity(2))
print(c*c == qt.Qobj(np.zeros((2,2))))
print(c.dag()*c.dag() == qt.Qobj(np.zeros((2,2))))

gamma1 = c.dag() + c
gamma2 = (c.dag() - c)/1j

# gamma1 == gamma1.dag(), gamma2 == gamma2.dag()
print(0.5*(gamma1 + 1j*gamma2) == c.dag())
print(0.5*(gamma1 - 1j*gamma2) == c)

print(anticommutator(gamma1, gamma2) == qt.Qobj(np.zeros((2,2))))
print(gamma1*gamma1 == qt.identity(2))
print(gamma2*gamma2 == qt.identity(2))
