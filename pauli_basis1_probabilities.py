import qutip as qt
from itertools import *
import numpy as np
import time
import matplotlib.pyplot as plt; plt.rcdefaults()

def pauli_basis(n):
	paulis = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
	names = ["".join(s) for s in product(["I", "X", "Y", "Z"], repeat=n)]
	P = [qt.tensor(*s) for s in product(paulis, repeat=n)]
	d = dict(zip(names, P))
	sizes = dict([(name, name.count("I")) for name in names])
	snames = sorted(zip(names, list(range(len(names)))), key=lambda x: x[0].count("I"), reverse=True)
	P2 = [P[sn[1]] for sn in snames]
	names2 = [sn[0] for sn in snames]
	starts = [0]
	now = names2[0].count("I")
	for i in range(len(names2)):
		if names2[i].count("I") != now:
			starts.append(i)
			now = names2[i].count("I")
	return names2, P2, d, dict(zip(list(range(len(starts))), starts))

def to_pauli(H):
	global pauli_op_names, pauli_ops, pbasis
	HP = [(qt.operator_to_vector(H).dag()*qt.operator_to_vector(p))[0][0][0] for p in pauli_ops]
	V = qt.Qobj(np.array(HP))/np.sqrt(len(pauli_ops))
	H2 = sum([h*pauli_ops[i] for i, h in enumerate(V.full().T[0])])
	return V

def disp(V):
	global pauli_op_names
	for i in range(V.shape[0]):
		print("%s: %f" % (pauli_op_names[i], V.full().T[0][i].real))

n = 2
dt = 0.08
pauli_op_names, pauli_ops, pbasis, starts = pauli_basis(n)

def size_winding_distribution(V):
	global pauli_op_names, pauli_ops, pbasis, starts
	d = []
	for l in range(len(starts)):
		if l+1 in starts:
			d.append(sum([v**2 for v in V[starts[l]:starts[l+1]]]))
		else:
			d.append(sum([v**2 for v in V[starts[l]:]]))
	return d

H = qt.rand_herm(2**n)
#H = qt.jmat((2**n -1)/2, 'z')
H.dims = [[2]*n, [2]*n]
U = (-1j*H*dt).expm()

O = qt.tensor(*[qt.sigmax() if i == n-1 else qt.identity(2) for i in range(n)])
plt.rc('font', size=8)  

fig, ax = plt.subplots()
V = to_pauli(O).full().T[0].real
bar = plt.bar(np.arange(len(pauli_op_names)), V)
#bar = plt.bar(np.arange(n+1), size_winding_distribution(V))
plt.xticks(np.arange(len(pauli_op_names)), pauli_op_names)
plt.ylim([-1,1])
while True:
	O = U.dag()*O*U
	V = to_pauli(O).full().T[0].real
	#disp(qt.Qobj(V))
	[b.set_height(v) for b, v in zip(bar,V.tolist())]
	#[b.set_height(v) for b, v in zip(bar,size_winding_distribution(V))]
	fig.canvas.draw()
	plt.pause(0.00001)
