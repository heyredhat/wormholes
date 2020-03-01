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
		print("%s: %s" % (pauli_op_names[i], V.full().T[0][i]))

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

#H = qt.rand_herm(2**n)
H = qt.jmat((2**n -1)/2, 'x')
H.dims = [[2]*n, [2]*n]
U = (-1j*H*dt).expm()

O = qt.tensor(*[qt.sigmax() if i == n-1 else qt.identity(2) for i in range(n)])
O2 = qt.tensor(*[qt.sigmay() if i == n-1 else qt.identity(2) for i in range(n)])
O3 = qt.tensor(*[qt.sigmaz() if i == n-1 else qt.identity(2) for i in range(n)])

#O = qt.rand_unitary(2**n)
O.dims = H.dims
O2.dims = H.dims
O3.dims = H.dims

plt.rc('font', size=8)  
fig, ax = plt.subplots()

POS_OR_MO = "momentum"

V = to_pauli(O).full().T[0] if POS_OR_MO == "momentum" else np.fft.ifft(to_pauli(O).full().T[0])
V = V/np.linalg.norm(V)

V2 = to_pauli(O2).full().T[0] if POS_OR_MO == "momentum" else np.fft.ifft(to_pauli(O2).full().T[0])
V2 = V2/np.linalg.norm(V2)

V3 = to_pauli(O3).full().T[0] if POS_OR_MO == "momentum" else np.fft.ifft(to_pauli(O3).full().T[0])
V3 = V3/np.linalg.norm(V3)

#p1 = plt.plot(np.arange(len(pauli_op_names)), V.real, color='#FFFF00')
#p2 = plt.plot(np.arange(len(pauli_op_names)), V.imag, color='#818100')
p3 = plt.fill_between(np.arange(len(pauli_op_names)), [(v*v.conj()).real for v in V], color='red')

#p4 = plt.plot(np.arange(len(pauli_op_names)), V2.real, color="#026117")
#p5 = plt.plot(np.arange(len(pauli_op_names)), V2.imag, color='#8CFFA5')
p6 = plt.fill_between(np.arange(len(pauli_op_names)), [(v*v.conj()).real for v in V2], color='green')

#p7 = plt.plot(np.arange(len(pauli_op_names)), V3.real, color="cyan")
#p8 = plt.plot(np.arange(len(pauli_op_names)), V3.imag, color='magenta')
p9 = plt.fill_between(np.arange(len(pauli_op_names)), [(v*v.conj()).real for v in V3], color='blue')

#bar = plt.bar(np.arange(n+1), size_winding_distribution(V))
plt.xticks(np.arange(len(pauli_op_names)), pauli_op_names)
plt.ylim([-1,1])
while True:
	#input()
	O = U.dag()*O*U
	O2 = U.dag()*O2*U
	O3 = U.dag()*O3*U

	V = to_pauli(O).full().T[0] if POS_OR_MO == "momentum" else np.fft.ifft(to_pauli(O).full().T[0])
	V = V/np.linalg.norm(V)

	V2 = to_pauli(O2).full().T[0] if POS_OR_MO == "momentum" else np.fft.ifft(to_pauli(O2).full().T[0])
	V2 = V2/np.linalg.norm(V2)

	V3 = to_pauli(O3).full().T[0] if POS_OR_MO == "momentum" else np.fft.ifft(to_pauli(O3).full().T[0])
	V3 = V3/np.linalg.norm(V3)
	#print(np.linalg.norm(V))
	#disp(qt.Qobj(V))
	plt.clf()

	#p1 = plt.plot(np.arange(len(pauli_op_names)), V.real)
	#p2 = plt.plot(np.arange(len(pauli_op_names)), V.imag, color='blue')
	p3 = plt.fill_between(np.arange(len(pauli_op_names)), [(v*v.conj()).real for v in V], color='red', alpha=0.5)

	#p4 = plt.plot(np.arange(len(pauli_op_names)), V2.real, color="green")
	#p5 = plt.plot(np.arange(len(pauli_op_names)), V2.imag, color='magenta')
	p6 = plt.fill_between(np.arange(len(pauli_op_names)), [(v*v.conj()).real for v in V2], color='yellow', alpha=0.5)

	#p7 = plt.plot(np.arange(len(pauli_op_names)), V3.real, color="cyan")
	#p8 = plt.plot(np.arange(len(pauli_op_names)), V3.imag, color='magenta')
	p9 = plt.fill_between(np.arange(len(pauli_op_names)), [(v*v.conj()).real for v in V3], color='blue', alpha=0.5)


	plt.ylim([-1,1])
	#[p1.set_height(v) for b, v in zip(bar,V.tolist())]
	#[b.set_height(v) for b, v in zip(bar,size_winding_distribution(V))]
	fig.canvas.draw()
	plt.pause(0.00001)
	#input()
