import time
import scipy
import qutip as qt
import numpy as np
from itertools import *
from functools import *
import scipy.sparse as sp
import matplotlib.pyplot as plt; plt.rcdefaults(); #plt.rc('font', size=8)

def xyz(dm):
    return [qt.expect(qt.sigmax(), dm),\
            qt.expect(qt.sigmay(), dm),\
            qt.expect(qt.sigmaz(), dm)]

def rand_scrambler(q, klocal=3):
    #if q == 2:
    #    Q = qt.rand_herm(4)
    #    Q.dims = [[2,2], [2,2]]
    #    return Q
    X = [qt.tensor(*[qt.sigmax() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
    Y = [qt.tensor(*[qt.sigmay() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]
    Z = [qt.tensor(*[qt.sigmaz() if i == j else qt.identity(2) for j in range(q)]) for i in range(q)]    
    return sum([reduce(lambda x, y: x*y,\
                    [np.random.randn(1)[0]*X[i] +\
                     np.random.randn(1)[0]*Y[i] +\
                     np.random.randn(1)[0]*Z[i]\
                             for i in p])
                                  for p in permutations(list(range(q)), klocal)])

###########################################################################

def perm_mat(Q, order):
    dims, perm = qt.permute._perm_inds(Q.dims[0], order)
    nzs = Q.data.nonzero()[0]
    wh = np.where(perm == nzs)[0]
    data = np.ones(len(wh), dtype=int)
    cols = perm[wh].T[0]
    perm_matrix = sp.coo_matrix((data, (wh, cols)),
                                shape=(Q.shape[0], Q.shape[0]), dtype=int)
    perm_matrix = qt.Qobj(perm_matrix.tocsr().todense())
    perm_matrix.dims = [Q.dims[0], Q.dims[0]]
    return perm_matrix

###########################################################################

def pauli_basis(n):
    print("constructing pauli basis for %d qubits..." % n)
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

###########################################################################

class Wormhole:
    def __init__(self, n=3, m=1, beta=0, g=np.pi, tL=10000, tR=10000, klocal=2, dt=0.08, construct_pauli_basis=True):
        self.n = n # n qubits for each black hole
        self.m = m # m message qubits for each black hole
        self.nwormhole_qubits = 2*self.n
        self.ntotal_qubits = self.nwormhole_qubits + 2*self.m
        self.beta = beta # beta = 1/temperature, so 0 = 1/Inf
        self.g = g # coupling time
        self.tL = tL # left black hole evolution time
        self.tR = tR # right black hole evolution time
        self.klocal = klocal # k for generating random scrambling hamiltonian for black hole
        self.dt = dt # for simulated time evolution
        self.construct()
        self.gscan() # scan for good g value
        if construct_pauli_basis:
            self.pauli_op_names, self.pauli_ops, self.pbasis, self.pstarts = pauli_basis(self.ntotal_qubits)

    ###########################################################################

    def construct(self, keep=False, evolve=False, callback=None, silent=False):
        if evolve:
            print("EVOLUTION")
        print("wormhole structure:") if not silent else None
        print("ref | msg ||| l_in | l_car || r_out | r_car") if not silent else None
        print("* "*self.m + "| " + "* "*self.m + "||| " + "* "*self.m + "| " + "* "*(self.n-self.m) + "|| " + "* "*self.m + "| " + "* "*(self.n-self.m)) if not silent else None
        print("constructing entangled reference and message...") if not silent else None
        self.REF_MSG = (qt.tensor(*[qt.basis(2,0) for i in range(2*self.m)]) + \
                        qt.tensor(*[qt.basis(2,1) for i in range(2*self.m)])).unit()
        print("constructing wormhole with %d qubits at each mouth..." % (self.n)) if not silent else None
        if not keep:
            print("constructing random %d-local scrambling hamiltonian..." % (self.klocal)) if not silent else None
        else:
            print("keeping existing hamiltonian...")
        self.H = rand_scrambler(self.n, klocal=self.klocal) if not keep else self.H
        self.H.dims = [[2]*self.n, [2]*self.n]
        print("constructing thermofield double state at inverse temp %.3f..." % (self.beta)) if not silent else None
        self.HL, self.HV = self.H.eigenstates()
        self.TFD = qt.tensor(self.REF_MSG, self.construct_TFD())
        if evolve:
            I = qt.identity(2**self.ntotal_qubits)
            I.dims = [[2]*self.ntotal_qubits, [2]*self.ntotal_qubits]
            callback(self.TFD, I, I)
        print("evolving left qubits backwards in time for time %.3f..." % (self.tL)) if not silent else None
        self.LBACK = qt.tensor(qt.identity(2**(2*self.m)), (1j*self.H*self.tL).expm(), qt.identity(2**self.n))
        self.LBACK.dims = [[2]*self.ntotal_qubits, [2]*self.ntotal_qubits]
        self.TFD2 = self.LBACK*self.TFD
        if evolve:
            for t in np.linspace(0, self.tL, int(1/self.dt)):
                LBACK = qt.tensor(qt.identity(2**(2*self.m)), (1j*self.H*t).expm(), qt.identity(2**self.n))
                LBACK.dims = [[2]*self.ntotal_qubits, [2]*self.ntotal_qubits]
                print("t: %.3f%%" % (100*t/self.tL))
                callback(LBACK*self.TFD, LBACK.dag(), LBACK)
        print("inserting message (%d qubit(s)) into left black hole..." % (self.m)) if not silent else None
        self.permuted = []
        for i in range(self.ntotal_qubits):
            if i >= self.m and i < 2*self.m:
                self.permuted.append(i+self.m)
            elif i >= 2*self.m and i < 3*self.m:
                self.permuted.append(i-self.m)
            else:
                self.permuted.append(i)
        self.TFD4 = self.TFD3 = self.TFD2.permute(self.permuted)
        if evolve:
            self.permutation_matrix = perm_mat(self.TFD2, self.permuted)
            callback(self.TFD4, self.permutation_matrix*self.LBACK.dag(), self.LBACK*self.permutation_matrix.dag())
        print("evolving left qubits forward in time for time %.3f..." % (self.tL)) if not silent else None
        self.LFORWARD = qt.tensor(qt.identity(2**(2*self.m)), (-1j*self.H*self.tL).expm(), qt.identity(2**self.n))
        self.LFORWARD.dims = [[2]*(self.ntotal_qubits), [2]*(self.ntotal_qubits)]
        self.TFD5 = self.LFORWARD*self.TFD4
        if evolve:
            for t in np.linspace(0, self.tL, int(1/self.dt)):
                LFORWARD = qt.tensor(qt.identity(2**(2*self.m)), (-1j*self.H*t).expm(), qt.identity(2**self.n))
                LFORWARD.dims = [[2]*self.ntotal_qubits, [2]*self.ntotal_qubits]
                print("t: %.3f%%" % (100*t/self.tL))
                callback(LFORWARD*self.TFD4, LFORWARD.dag()*self.permutation_matrix*self.LBACK.dag(), self.LBACK*self.permutation_matrix.dag()*LFORWARD)
        print("constructing coupling...") if not silent else None
        self.ncarrier_qubits = self.n-self.m
        self.carrier_indices = [(3*self.m+i, 3*self.m+i+self.n) for i in range(self.ncarrier_qubits)]
        terms = []
        for carrier_pair in self.carrier_indices:
            ZL = qt.tensor(*[qt.sigmaz() if i == carrier_pair[0] else qt.identity(2) for i in range(self.ntotal_qubits)])
            ZR = qt.tensor(*[qt.sigmaz() if i == carrier_pair[1] else qt.identity(2) for i in range(self.ntotal_qubits)])
            terms.append(ZL*ZR)
        self.V = (1/self.ncarrier_qubits)*sum(terms)
        print("evolving carrier qubits with coupling for time g = %.3f..." % self.g) if not silent else None
        self.COUPLING = (1j*self.g*self.V).expm()
        self.TFD6 = self.COUPLING*self.TFD5
        if evolve:
            for t in np.linspace(0, self.g, int(1/self.dt)):
                COUPLING = (1j*t*self.V).expm()
                print("t: %.3f%%" % (100*t/self.g))
                callback(COUPLING*self.TFD5, COUPLING.dag()*self.LFORWARD.dag()*self.permutation_matrix*self.LBACK.dag(), self.LBACK*self.permutation_matrix.dag()*self.LFORWARD*COUPLING)
        print("evolving right qubits forward in time for time %.3f..." % (self.tR)) if not silent else None
        self.RFORWARD = qt.tensor(qt.identity(2**(2*self.m)), qt.identity(2**self.n), (-1j*self.H.trans()*self.tR).expm())
        self.RFORWARD.dims = [[2]*(self.ntotal_qubits), [2]*(self.ntotal_qubits)]
        self.FINAL = self.RFORWARD*self.TFD6
        if evolve:
            for t in np.linspace(0, self.tR, int(1/self.dt)):
                RFORWARD = qt.tensor(qt.identity(2**(2*self.m)), qt.identity(2**self.n), (-1j*self.H.trans()*t).expm())
                RFORWARD.dims = [[2]*self.ntotal_qubits, [2]*self.ntotal_qubits]
                print("t: %.3f%%" % (100*t/self.tR))
                callback(RFORWARD*self.TFD6, RFORWARD.dag()*self.COUPLING.dag()*self.LFORWARD.dag()*self.permutation_matrix*self.LBACK.dag(), self.LBACK*self.permutation_matrix.dag()*self.LFORWARD*self.COUPLING*RFORWARD)
        self.output_index = list(range(2*self.m+self.n, 3*self.m+self.n))
        print("done!") if not silent else None
        self.calculate_mutual_information()
        print("mutual information between reference and output: %.5f (should be: %.5f)" % (self.mutual_info, self.should_be)) if not silent else None

    ###########################################################################

    def construct_TFD(self):
        return (1/np.sqrt((-self.beta*self.H).expm().tr()))*\
                 sum([np.exp(-(1/2)*self.beta*l)*\
                     qt.tensor(self.HV[j].conj(), self.HV[j])\
                        for j, l in enumerate(self.HL)])

    ###########################################################################

    def calculate_mutual_information(self, loud=False):
        self.should_be = 2*np.log(2)
        self.FIN = self.FINAL.ptrace(list(range(self.m))+list(range(2*self.m + self.n, 3*self.m + self.n)))
        self.mutual_info = qt.entropy_mutual(self.FIN, list(range(self.m)), list(range(self.m, 2*self.m)))

        if loud:
            for i in range(self.m, self.ntotal_qubits):
                fin = self.FINAL.ptrace(list(range(self.m))+[i])
                mut_inf = qt.entropy_mutual(fin, list(range(self.m)), [self.m])
                print("%s\te between ref and qubit %d: %.5f" % ("*" if i in self.output_index else "", i, mut_inf))
            print()
    
    ###########################################################################

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
        for g in np.linspace(0.0001, 10, n):
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

    ###########################################################################

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
    
    ###########################################################################

    def jump(self, guy):
        proj = qt.tensor(guy*guy.dag(), *[qt.identity(2) for i in range(self.ntotal_qubits-self.m)])
        return (proj*self.FINAL).unit().ptrace(self.output_index)

    ###########################################################################

    def to_op_basis(self, H):
        HP = [(qt.operator_to_vector(H).dag()*qt.operator_to_vector(p))[0][0][0] for p in self.pauli_ops]
        V = qt.Qobj(np.array(HP))/np.sqrt(len(self.pauli_ops))
        #H2 = sum([h*self.pauli_ops[i] for i, h in enumerate(V.full().T[0])])
        return V

    def pdisp(self, V):
        for i in range(V.shape[0]):
            print("%s: %s" % (self.pauli_op_names[i], V.full().T[0][i]))

    def size_winding_distribution(self, op):
        V = self.to_op_basis(op)
        d = []
        for l in range(len(self.pstarts)):
            if l+1 in self.pstarts:
                d.append(sum([v**2 for v in V[self.pstarts[l]:self.pstarts[l+1]]]))
            else:
                d.append(sum([v**2 for v in V[self.pstarts[l]:]]))
        return qt.Qobj(np.array(d))

    def watch_operator(self, o):
        fig, ax = plt.subplots()
        def __callback__(state, left, right):
            V = self.to_op_basis(left*o*right).full().T[0]
            plt.clf()
            p = plt.fill_between(np.arange(len(self.pauli_op_names)), [(v*v.conj()).real for v in V], color='red')
            plt.ylim([0,1])
            #plt.xlim([0, 50])
            #plt.plot(np.arange(len(w.pauli_op_names)), [1 if name=="IIZIII"\
            #                                              #or name=="IIYIIIII"\
            #                                              #or name=="IIZIIIII"                                                
            #                                             else 0 for name in w.pauli_op_names], color='blue', alpha=0.5)
            #   
            #plt.plot(np.arange(len(w.pauli_op_names)), [1 if (name=="IIIIZI"\
            #                                              or name=="IIIIYI"\
            #                                              or name=="IIIIXI")                                                
            #                                             else 0 for name in w.pauli_op_names], color='green', alpha=0.5)
            #
            fig.canvas.draw()
            plt.pause(0.00001)
            #input()
        self.construct(evolve=True,\
                       keep=True,\
                       silent=False,\
                       callback=__callback__)

    def watch_operator_size(self, o):
        fig, ax = plt.subplots()
        def __callback__(state, left, right):
            V = self.size_winding_distribution(left*o*right).full().T[0]
            plt.clf()
            p = plt.fill_between(np.arange(len(V)), [v.real for v in V], color='red')
            plt.ylim([0,1])
            fig.canvas.draw()
            plt.pause(0.00001)
            #input()
        self.construct(evolve=True,\
                       keep=True,\
                       silent=False,\
                       callback=__callback__)
###########################################################################

w = Wormhole(n=7, m=1, beta=0, g=np.pi, tL=10000, tR=10000,\
            klocal=4, construct_pauli_basis=False, dt=0.08)

###########################################################################

#O = qt.tensor(*[qt.sigmaz() if i == 2 else qt.identity(2) for i in range(w.ntotal_qubits)])
#w.watch_operator_size(O)

###########################################################################