import time
import scipy
import qutip as qt
import numpy as np
from itertools import *
from functools import *

def rand_scrambler(q, klocal=3):
	X = [qt.tensor(*[qt.sigmax() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
	Y = [qt.tensor(*[qt.sigmay() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
	Z = [qt.tensor(*[qt.sigmaz() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]	
	return sum([reduce(lambda x, y: x*y,\
					[np.random.randn(1)[0]*X[i] +\
					 np.random.randn(1)[0]*Y[i] +\
					 np.random.randn(1)[0]*Z[i]\
					 		for i in p])
				 	 			for p in permutations(list(range(q)), klocal)])

class Wormhole:
	def __init__(self, n=3, m=1, beta=0, g=np.pi, tL=10000, tR=10000, klocal=2):
		self.n = n # n qubits for each black hole
		self.m = m # m message qubits for each black hole
		self.nwormhole_qubits = 2*self.n
		self.ntotal_qubits = self.nwormhole_qubits + 2*self.m
		self.beta = beta # beta = 1/temperature, so 0 = 1/Inf
		self.g = g # coupling time
		self.tL = tL # left black hole evolution time
		self.tR = tR # right black hole evolution time
		self.klocal = klocal # k for generating random scrambling hamiltonian for black hole
		self.construct()
		self.gscan() # scan for good g value

	def construct(self):
		print("constructing wormhole with %d qubits at each mouth..." % (self.n))
		print("constructing random %d-local scrambling hamiltonian..." % (self.klocal))
		self.H = rand_scrambler(self.n)
		self.H.dims = [[2]*self.n, [2]*self.n]
		print("constructing thermofield double state at inverse temp %.3f..." % (self.beta))
		self.HL, self.HV = self.H.eigenstates()
		self.TFD = self.construct_TFD()
		print("evolving left qubits backwards in time for time %.3f..." % (self.tL))
		self.LBACK = qt.tensor((1j*self.H*self.tL).expm(), qt.identity(2**self.n))
		self.LBACK.dims = [[2]*self.nwormhole_qubits, [2]*self.nwormhole_qubits]
		self.TFD2 = self.LBACK*self.TFD
		print("inserting message (%d qubit(s)) into left black hole..." % (self.m))
		self.REF_MSG = (qt.tensor(*[qt.basis(2,0) for i in range(2*self.m)]) + \
						qt.tensor(*[qt.basis(2,1) for i in range(2*self.m)])).unit()
		self.TFD3 = qt.tensor(self.REF_MSG, self.TFD2)
		self.permuted = []
		for i in range(self.ntotal_qubits):
			if i >= self.m and i < 2*self.m:
				self.permuted.append(i+self.m)
			elif i >= 2*self.m and i < 3*self.m:
				self.permuted.append(i-self.m)
			else:
				self.permuted.append(i)
		self.TFD4 = self.TFD3.permute(self.permuted)
		print("evolving left qubits forward in time for time %.3f..." % (self.tL))
		self.LFORWARD = qt.tensor(qt.identity(2**(2*self.m)), (-1j*self.H*self.tL).expm(), qt.identity(2**self.n))
		self.LFORWARD.dims = [[2]*(self.ntotal_qubits), [2]*(self.ntotal_qubits)]
		self.TFD5 = self.LFORWARD*self.TFD4
		print("constructing coupling...")
		self.ncarrier_qubits = self.n-self.m
		self.carrier_indices = [(3*self.m+i, 3*self.m+i+self.n) for i in range(self.ncarrier_qubits)]
		terms = []
		for carrier_pair in self.carrier_indices:
			ZL = qt.tensor(*[qt.sigmaz() if i == carrier_pair[0] else qt.identity(2) for i in range(self.ntotal_qubits)])
			ZR = qt.tensor(*[qt.sigmaz() if i == carrier_pair[1] else qt.identity(2) for i in range(self.ntotal_qubits)])
			terms.append(ZL*ZR)
		self.V = (1/self.ncarrier_qubits)*sum(terms)
		print("evolving carrier qubits with coupling for time g = %.3f..." % self.g)
		self.COUPLING = (1j*self.g*self.V).expm()
		self.TFD6 = self.COUPLING*self.TFD5
		print("evolving right qubits forward in time for time %.3f..." % (self.tR))
		self.RFORWARD = qt.tensor(qt.identity(2**(2*self.m)), qt.identity(2**self.n), (-1j*self.H.trans()*self.tR).expm())
		self.RFORWARD.dims = [[2]*(self.ntotal_qubits), [2]*(self.ntotal_qubits)]
		self.FINAL = self.RFORWARD*self.TFD6
		self.output_index = 2*self.m+self.n
		print("done!")
		self.calculate_mutual_information()
		print("mutual information between reference and output: %.5f (should be: %.5f)" % (self.mutual_info, self.should_be))

	def construct_TFD(self):
		return (1/np.sqrt((-self.beta*self.H).expm().tr()))*\
				 sum([np.exp(-(1/2)*self.beta*l)*\
		 			qt.tensor(self.HV[j].conj(), self.HV[j])\
						for j, l in enumerate(self.HL)])

	def calculate_mutual_information(self, loud=True):
		self.should_be = 2*np.log(2)
		self.FIN = self.FINAL.ptrace(list(range(self.m))+list(range(2*self.m + self.n, 3*self.m + self.n)))
		self.mutual_info = qt.entropy_mutual(self.FIN, list(range(self.m)), list(range(self.m, 2*self.m)))

		if loud:
			for i in range(self.m, self.ntotal_qubits):
				fin = self.FINAL.ptrace(list(range(self.m))+[i])
				mut_inf = qt.entropy_mutual(fin, list(range(self.m)), [self.m])
				print("%s\te between ref and qubit %d: %.5f" % ("*" if i == self.output_index else "", i, mut_inf))

	def rerun(self, g):
		self.g = g
		self.COUPLING = (1j*self.g*self.V).expm()
		self.TFD6 = self.COUPLING*self.TFD5
		self.FINAL = self.RFORWARD*self.TFD6
		self.calculate_mutual_information()
		print("g: %.5f | e: %.5f | desired: %.5f" % (self.g, self.mutual_info, self.should_be))
	
	def gscan(self, n=50, teleport=False):
		print("gscanning...")
		max_mutual_info = -1
		max_g = 0
		for g in np.linspace(0, 10, n):
			if teleport:
				self.teleport(silent=True)
			else:
				self.rerun(g)
			if self.mutual_info > max_mutual_info:
				max_mutual_info = self.mutual_info
				max_g = self.g
			elif self.mutual_info < max_mutual_info:
				if not teleport:
					break
		print("setting g to %.5f with mutual info %.5f..." % (max_g, max_mutual_info))
		self.g = max_g

	def measure(self, i, op): # measures on TFD5
		L, V = op.eigenstates()
		substate = self.TFD5.ptrace(i)
		probabilities = np.array([(substate*V[j]*V[j].dag()).tr() for j in range(len(V))])
		probabilities = probabilities/sum(probabilities)
		choice = np.random.choice(list(range(len(L))), p=probabilities)
		projector = qt.tensor(*[V[choice]*V[choice].dag() if j == i else qt.identity(2) for j in range(self.ntotal_qubits)])
		self.TFD5 = (projector*self.TFD5).unit()
		return L[choice]

	def teleport(self, silent=False):
		print("after reverse L/insert msg/forward L, instead of quantum coupling...") if not silent else None
		self.TFD5 = self.LFORWARD*self.TFD4
		print("measuring left carrier qubits in the z-basis...") if not silent else None
		self.measurements = [self.measure(i, qt.sigmaz()) for i in range(3*self.m, 3*self.m+self.ncarrier_qubits)]
		print("\tresults: %s" % self.measurements)
		print("applying right unitary correction...") if not silent else None
		self.correction = (1j*self.g*sum([z*qt.tensor(*[qt.sigmaz() if j == 3*self.m+i+self.n else qt.identity(2) for j in range(self.ntotal_qubits)])/(self.n-self.m) for i, z in enumerate(self.measurements)])).expm()
		self.TFD6 = self.correction*self.TFD5
		print("evolving right qubits forward in time for time %.3f..." % (self.tR)) if not silent else None
		self.FINAL = self.RFORWARD*self.TFD6
		print("done!") if not silent else None
		self.calculate_mutual_information()
		print("g: %.5f | e: %.5f | desired: %.5f" % (self.g, self.mutual_info, self.should_be))
	
w = Wormhole(n=7, m=1, beta=0, g=np.pi, tL=10000, tR=10000, klocal=4)