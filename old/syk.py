import qutip as qt
import numpy as np
from itertools import *
from functools import *
import math

####################################################

def make_majoranas(n):
	psiL = {}
	psiR = {}
	for j in range(1, int(n/2)+1):
		psiL[2*j-1] = (1/2**(n/2))*(1/np.sqrt(2))*\
				qt.tensor(*([qt.sigmaz(), qt.sigmax()]*int(j-1) +\
				[qt.sigmax(), qt.sigmax()] + \
				[qt.identity(2), qt.identity(2)]*int(n/2-j)))
		psiR[2*j-1] = (1/2**(n/2))*(1/np.sqrt(2))*\
				qt.tensor(*([qt.sigmaz(), qt.sigmax()]*int(j-1) +\
				[qt.identity(2), qt.sigmay()] + \
				[qt.identity(2), qt.identity(2)]*int(n/2-j)))
		psiL[2*j] = (1/2**(n/2))*(1/np.sqrt(2))*\
				qt.tensor(*([qt.sigmaz(), qt.sigmax()]*int(j-1) +\
				[qt.sigmay(), qt.sigmax()] + \
				[qt.identity(2), qt.identity(2)]*int(n/2-j)))
		psiR[2*j] = (1/2**(n/2))*(1/np.sqrt(2))*\
				qt.tensor(*([qt.sigmaz(), qt.sigmax()]*int(j-1) +\
				[qt.identity(2), qt.sigmaz()] + \
				[qt.identity(2), qt.identity(2)]*int(n/2-j)))
	return psiL, psiR

####################################################

def make_fermions(psiL, psiR):
	creates = []
	destroys = []
	for i in range(len(psiL)):
		creates.append(0.5*(psiL[i+1] + 1j*psiR[i+1]))
		destroys.append(0.5*(psiL[i+1] - 1j*psiR[i+1]))
	numbers = [creates[i]*destroys[i] for i in range(len(psiL))]
	return {"creates": creates, "destroys": destroys, "numbers": numbers}

####################################################

n = 8
psiL, psiR = make_majoranas(n)
fermions = make_fermions(psiL, psiR)

####################################################

def majorana_expectations(state):
	global psiL, psiR
	state.dims = psiL[1].dims
	left = [qt.expect(psiL[i+1]*psiL[i+1], state) for i in range(len(psiL))]
	right = [qt.expect(psiR[i+1]*psiR[i+1], state) for i in range(len(psiL))]
	print("majorana # expectations:")
	print("L:")
	for i in range(len(psiL)):
		print("m%d : %s" % (i, left[i]))
	print("R:")
	for i in range(len(psiL)):
		print("m%d : %s" % (i, right[i]))
	#return left, right

def fermion_expectations(state):
	global psiL, psiR, fermions
	state.dims = psiL[1].dims
	nums = [qt.expect(fermions["numbers"][i], state) for i in range(len(fermions["numbers"]))]
	print("fermion # expectations:")
	for i in range(len(nums)):
		print("f%d : %s" % (i, nums[i]))

####################################################

def make_syk(J=1, q=4): # q-local
	global psiL, psiR
	d = len(psiL)
	var = (2**(q-1))*(J**2)*math.factorial(q-1)\
			/(q*n**(q-1))
	combins = list(combinations(list(range(d)), 4))
	H_coeffs = np.random.normal(scale=var, size=len(combins))
	HL = ((1j)**(q/2))*sum([H_coeffs[i]*reduce(lambda x, y: x*y, [psiL[c+1] for c in C]) for i, C in enumerate(combins)])
	HR = ((1j)**(q/2))*sum([H_coeffs[i]*reduce(lambda x, y: x*y, [psiR[c+1] for c in C]) for i, C in enumerate(combins)]).trans()
	H = HL + HR
	return HL, HR, H

####################################################

HL, HR, H = make_syk(J=1, q=4)

####################################################

def commutator(A, B):
	return A*B - B*A
	
def anticommutator(A, B):
	return A*B + B*A

def test_majoranas1():
	global psiL
	print("psiL")
	for pair in product(psiL.keys(), repeat=2):
		anti = anticommutator(psiL[pair[0]], psiL[pair[1]]).tr()
		print("%s: %s"% (pair, anti))
	print("psiR")
	for pair in product(psiR.keys(), repeat=2):
		anti = anticommutator(psiR[pair[0]], psiR[pair[1]]).tr()
		print("%s: %s"% (pair, anti))

####################################################

state = qt.rand_ket(2**n)
state.dims = [[2]*n, [1]*n]