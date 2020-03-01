import qutip as qt
import numpy as np
from itertools import *
import matplotlib.pyplot as plt; plt.rcdefaults(); plt.rc('font', size=8)
import networkx as nx
from scipy.linalg import eigh

############################################################################

def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

def diff_by_one(a, b):
    return diff_letters(a, b) == 1

###########################################################################

class PauliBasis:
    def __init__(self, n):
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
        self.dim = len(names2)
        self.names = names2
        self.ops = P2
        self.basis = d
        self.starts = dict(zip(list(range(len(starts))), starts))
        self.adjacency = qt.Qobj(np.array([[1 if diff_by_one(self.names[i], self.names[j]) else 0 for j in range(self.dim)] for i in range(self.dim)]))
        self.degree = qt.Qobj(np.diag(np.array([sum(self.adjacency[i][0]) for i in range(self.dim)])))
        self.laplacian = self.degree - self.adjacency
        self.ilaplacian = qt.Qobj(np.linalg.inv(self.laplacian.full()))
        self.L, self.V = self.laplacian.eigenstates()
        self.iL, self.iV = self.ilaplacian.eigenstates()

    ###########################################################################

    def to_pauli(self, H):
        H.dims = self.ops[0].dims
        HP = [(qt.operator_to_vector(H).dag()*qt.operator_to_vector(p))[0][0][0] for p in self.ops]
        return qt.Qobj(np.array(HP))/np.sqrt(len(self.ops))
        
    def from_pauli(self, V):
        return sum([h*self.ops[i] for i, h in enumerate(V.full().T[0])])

    ###########################################################################

    def disp(self, V):
        for i in range(V.shape[0]):
            print("%s: %s" % (self.names[i], V.full().T[0][i]))

    ###########################################################################

    def to_laplace(self, H):
        V = self.to_pauli(H)
        return qt.Qobj(np.array([(V.dag()*v)[0][0][0] for v in self.V]))

    def from_laplace(self, L):
        return self.from_pauli(sum([l*self.V[i] for i, l in enumerate(L.full().T[0])]))

    ###########################################################################

    def show_graph(self):
        #V = np.array([v.full().T[0] for v in self.V])
        #x = V[:,1] 
        #y = V[:,2]
        #spectral_coordinates = {i : (x[i], y[i]) for i in range(n)}
        plt.clf()
        G = nx.from_numpy_matrix(self.adjacency.full().real)
        pos = nx.spectral_layout(G, scale=10)
        nx.draw_networkx_nodes(G, pos,\
                       node_color='r',\
                       node_size=700,\
                       alpha=0.8)
        nx.draw_networkx_edges(G,\
                            pos,\
                            width=1.0, alpha=0.5)
        labels = dict(zip(list(range(self.dim)), self.names))
        nx.draw_networkx_labels(G, pos, labels, font_size=13, opacity=0.6)
        plt.show()

    ###########################################################################

    def viz_op(self, op, fig, ax):
        probabilities = np.array([(v*v.conj()).real for v in self.to_pauli(op).full().T[0]])
        plt.clf()
        #ax.clear()
        G = nx.from_numpy_matrix(self.adjacency.full().real)
        pos = nx.spectral_layout(G, scale=50)
        nx.draw_networkx_nodes(G, pos,\
                       node_color='r',\
                       node_size=750,\
                       alpha=probabilities)
        nx.draw_networkx_edges(G,\
                            pos,\
                            width=0.2, alpha=0.5)
        labels = dict(zip(list(range(self.dim)), self.names))
        nx.draw_networkx_labels(G, pos, labels, font_size=12, alpha=0.5)
        fig.canvas.draw()
        plt.pause(0.00001)

    def viz_laplace(self, op, fig, ax):
        L = self.to_laplace(op).full().T[0]
        plt.clf()
        p1 = plt.plot(np.arange(self.dim), L.real, color='#FFFF00')
        p2 = plt.plot(np.arange(self.dim), L.imag, color='#818100')
        p3 = plt.fill_between(np.arange(self.dim), [(v*v.conj()).real for v in L], color='red')
        plt.xticks(np.arange(self.dim), ["%.0f" % l for l in self.L])
        plt.ylim([-1,1])
        fig.canvas.draw()
        plt.pause(0.00001)

    ###########################################################################

    def loop_viz_op(self, o, U):
        fig, ax = plt.subplots()
        while True:
            P.viz_op(o, fig, ax)
            #input()
            o = U.dag()*o*U

    def loop_viz_laplace(self, o, U):
        fig, ax = plt.subplots()
        while True:
            P.viz_laplace(o, fig, ax)
            #input()
            o = U.dag()*o*U

###########################################################################

n = 2
dt = 0.08
P = PauliBasis(n)

###########################################################################

H = qt.rand_herm(2**n) #qt.jmat((2**n -1)/2, 'z')
H.dims = [[2]*n, [2]*n]
U = (-1j*dt*H).expm()

###########################################################################

O = qt.tensor([qt.sigmax()\
        if i == 0 else qt.identity(2) for i in range(n)])

###########################################################################

P.loop_viz_op(O, U)
#P.loop_viz_laplace(O, U)


