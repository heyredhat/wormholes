import qutip as qt
import numpy as np
from itertools import *
from functools import *
import scipy

def xyz(dm):
	return [qt.expect(qt.sigmax(), dm),\
			qt.expect(qt.sigmay(), dm),\
			qt.expect(qt.sigmaz(), dm)]

def make_TFD(H, beta=1):
	L, V = H.eigenstates()
	return\
		(1/np.sqrt((-beta*H).expm().tr()))*\
			sum([np.exp(-(1/2)*beta*l)*\
	 			qt.tensor(V[j].conj(), V[j])\
					for j, l in enumerate(L)])

def rand_scrambler(q,klocal=3):
	#a = [qt.tensor(*[qt.destroy(2) if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
	X = [qt.tensor(*[qt.sigmax() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
	Y = [qt.tensor(*[qt.sigmay() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
	Z = [qt.tensor(*[qt.sigmaz() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
	everyone = X+Y+Z
	#return sum(np.random.randn(1)[0]*reduce(lambda x,y:x*y, p) for p in permutations(everyone, r=2))
	
	return sum([reduce(lambda x, y: x*y,\
		[np.random.randn(1)[0]*X[i] + np.random.randn(1)[0]*Y[i] + np.random.randn(1)[0]*Z[i]\
		 		for i in p])
	 	 			for p in permutations(list(range(q)), klocal)])
	#gamma1 = [aa.dag() + aa for aa in a]
	#gamma2 = [(aa.dag() - aa)/1j for aa in a]
	#return sum([np.random.randn(1)[0]*reduce(lambda x,y: x*y, perm) for perm in permutations(a, 2)])#+\
			#sum([np.random.randn(1)[0]*reduce(lambda x,y: x*y, perm) for perm in permutations(gamma2, 2)])

def rand_scrambler1(q):
	return 0.5*qt.Qobj(np.array([[-1, 0, 0, -1, 0, -1, -1, 0],\
								 [0, 1, -1, 0, -1, 0, 0, 1],\
								 [0, -1, 1, 0, -1, 0, 0, 1],\
								 [1, 0, 0, 1, 0, -1, -1, 0],\
								 [0, -1, -1, 0, 1, 0, 0, 1],\
								 [1, 0, 0, -1, 0, 1, -1, 0],\
								 [1, 0, 0, -1, 0, -1, 1, 0],\
								 [0, -1, -1, 0, -1, 0, 0, -1]]))

tL = 100000
tR = 100000
temp = float("Inf")
beta = 0
g = np.pi
m = 1
n = 9

print("loading...")
q = 2*n
total_q = 2*n + 2
H = rand_scrambler(n)
H.dims = [[2]*n, [2]*n]
HL, HV = H.eigenstates()
TFD = make_TFD(H, beta=beta)
LBACK = qt.tensor((1j*H*tL).expm(), qt.identity(2**n))
# Evolve left qubits backwards in time
#LBACK = qt.tensor(U.dag(), qt.identity(2**n))
LBACK.dims = [[2]*q, [2]*q]
TFD2 = LBACK*TFD
# Insert message
ref_msg = qt.bell_state('00')
TFD3 = qt.tensor(ref_msg, TFD2)
#pairs = [(1+i, 1+i+m) for i in range(m)]
#TFD4 = qt.tensor_swap(TFD3, *pairs)
TFD4 = TFD3.permute([i if i != 1 and i != 2 else 2 if i == 1 else 1 for i in range(q+2)])
#print([i if i != 1 and i != 2 else 2 if i == 1 else 1 for i in range(q+2)])
# Evolve left qubits forward in time
LFORWARD = qt.tensor(qt.identity(2), qt.identity(2**m), (-1j*H*tL).expm(), qt.identity(2**n))
#LFORWARD = qt.tensor(qt.identity(2), qt.identity(2**m), U, qt.identity(2**n))
LFORWARD.dims = [[2]*(q+m+1), [2]*(q+m+1)]
TFD5 = LFORWARD*TFD4
# Couple
c = n-m
cind = [(2*m + i +1, 2*m+n+i +1) for i in range(c)]
terms = []
for carrier_pair in cind:
	ZL = qt.tensor(*[qt.identity(2) if i != carrier_pair[0] else qt.sigmaz() for i in range(q+m+1)])
	ZR = qt.tensor(*[qt.identity(2) if i != carrier_pair[1] else qt.sigmaz() for i in range(q+m+1)])
	terms.append(ZL*ZR)
V = (1/(n-m))*sum(terms)
# ...
# Evolve right qubits forward in time
RFORWARD = qt.tensor(qt.identity(2), qt.identity(2**m), qt.identity(2**n), (-1j*H.trans()*tR).expm())
#RFORWARD = qt.tensor(qt.identity(2), qt.identity(2**m), qt.identity(2**n), U.trans())
RFORWARD.dims = [[2]*(q+m+1), [2]*(q+m+1)]

def WORMHOLE(tL, tR, beta, g, m, n):
	#H = qt.rand_herm(2**n)
	#H = rand_scrambler(n)
	global q, H, HL, HV, TFD, LBACK, TFD2, TFD3
	global TFD4, LFORWARD, TFD5, V
	#H = 1j*qt.Qobj(scipy.linalg.logm(U.full()))
	
	COUPLING = (1j*g*V).expm()
	TFD6 = COUPLING*TFD5
	TFD7 = RFORWARD*TFD6
	FIN = TFD7
	o = m+n+1
	#print("%d should be %f" % (o, 2*np.log(2)))
	if False:
		for i in range(len(FIN.dims[0])):
			if i != 0:
				if i == o:
					print("*")
				f = FIN.ptrace((0, i))
				e = qt.entropy_mutual(f, 0, 1)
				print ("subs 0 and %d: e = %.6f" % (i,e))
		print()

	fin = FIN.ptrace((0, o))
	e = qt.entropy_mutual(fin, 0, 1)
	should_be = 2*np.log(2)
	return "e: %f | %f" % (e, should_be)

if True:
	for i in np.linspace(0, 10, 50):
		print("g = %f | %s" % (i, WORMHOLE(tL, tR, beta, i, m, n)))
		#print("g = %f" % i)# | %s" % (i, WORMHOLE(tL, tR, beta, i, m, n)))
		#WORMHOLE(tL, tR, beta, i, m, n)
