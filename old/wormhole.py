import qutip as qt
import numpy as np

def xyz(dm):
	return [qt.expect(qt.sigmax(), dm),\
			qt.expect(qt.sigmay(), dm),\
			qt.expect(qt.sigmaz(), dm)]

tL = 1000
tR = 1000
temp = float("Inf")
beta = 0
g = np.pi

m = 1
msg = qt.rand_ket(2)

n = 6
q = 2*n
H = qt.rand_herm(2**n)
H.dims = [[2]*n, [2]*n]
HL, HV = H.eigenstates()

def make_TFD(H, beta=1):
	L, V = H.eigenstates()
	return\
		(1/np.sqrt((-beta*H).expm().tr()))*\
			sum([np.exp(-(1/2)*beta*l)*\
	 			qt.tensor(V[j].conj(), V[j])\
					for j, l in enumerate(L)])

TFD = make_TFD(H, beta=beta)

# Evolve left qubits backwards in time
LBACK = qt.tensor((1j*H*tL).expm(), qt.identity(2**n))
LBACK.dims = [[2]*q, [2]*q]
TFD2 = LBACK*TFD
# Insert message
TFD3 = qt.tensor(msg, TFD2)
pairs = [(i, i+m) for i in range(m)]
TFD4 = qt.tensor_swap(TFD3, *pairs)
# Evolve left qubits forward in time
LFORWARD = qt.tensor(qt.identity(2**m), (-1j*H*tL).expm(), qt.identity(2**n))
LFORWARD.dims = [[2]*(q+m), [2]*(q+m)]
TFD5 = LFORWARD*TFD4
# Couple
c = n-m
cind = [(2*m + i, 2*m+n+i) for i in range(c)]
terms = []
for carrier_pair in cind:
	ZL = qt.tensor(*[qt.identity(2) if i != carrier_pair[0] else qt.sigmaz() for i in range(q+m)])
	ZR = qt.tensor(*[qt.identity(2) if i != carrier_pair[1] else qt.sigmaz() for i in range(q+m)])
	terms.append(ZL*ZR)
V = (1/(n-m))*sum(terms)
COUPLING = (1j*g*V).expm()
TFD6 = COUPLING*TFD5
# Evolve right qubits forward in time
RFORWARD = qt.tensor(qt.identity(2**m), qt.identity(2**n), (-1j*H.trans()*tR).expm())
RFORWARD.dims = [[2]*(q+m), [2]*(q+m)]
TFD7 = RFORWARD*TFD6

FIN = TFD7
o = m+n

print(msg*msg.dag())
print(FIN.ptrace(o))
print(xyz(msg*msg.dag()))
print(xyz(FIN.ptrace(o)))