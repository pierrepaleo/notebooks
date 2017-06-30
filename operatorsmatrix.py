import numpy as np
from math import pi, sqrt
import pywt
from utils import gradient

def build_wavelet_matrix_1d(n, wname="haar"):
	inp = np.zeros(n)
	inp[0] = 1
	res = np.zeros((n, n))
	for i in range(n):
		DWT = pywt.wavedec(inp, wname, mode="periodization")
		pos_j = 0
		c = DWT[0].ravel()
		L = c.shape[0]
		res[:L, i] = DWT[0].ravel()
		pos_j += L
		for j in range(1, len(DWT)):
			coeffs = DWT[j].ravel()
			L = coeffs.shape[0]
			res[pos_j:pos_j+L, i] = coeffs
			pos_j += L
		inp = np.roll(inp, 1)
	return res



def build_wavelet_matrix_2d(n, wname="haar"):
	inp = np.zeros((n, n))
	res = np.zeros((n**2, n**2))
	for i in range(n):
		for j in range(n):
			I = i*n+j # "ravel" index
			inp[i,j] = 1
			DWT = pywt.wavedec2(inp, wname, mode="periodization")
			c = DWT[0].ravel()
			pos_k = 0
			L = c.shape[0]
			res[:L, I] = c
			pos_k += L
			for k in range(1, len(DWT)):
				coeffs_H = DWT[k][0].ravel()
				coeffs_V = DWT[k][1].ravel()
				coeffs_D = DWT[k][2].ravel()
				coeffs = np.array([coeffs_H, coeffs_V, coeffs_D]).ravel()
				L = coeffs.shape[0]
				res[pos_k:pos_k+L, I] = coeffs
				pos_k += L
			inp[i,j] = 0
	return res



"""
from scipy.misc import ascent; img = ascent();
u = img[50:54, 50:54]
D2 = pywt.wavedec2(u, "haar", mode="per", level=2)
D2
Out[252]: 
[array([[ 357.25]]),
 (array([[-1.75]]), array([[-0.75]]), array([[-0.75]])),
 (array([[ 3. , -1.5],
         [-2.5, -0.5]]), array([[ 0. , -0.5],
         [ 0.5, -0.5]]), array([[ 0. ,  1.5],
         [ 0.5, -0.5]]))]

build_wavelet_matrix_2d(4).dot(u.ravel()).reshape((4,4))
array([[ 357.25,   -1.75,   -0.75,   -0.75],
       [   3.  ,   -1.5 ,   -2.5 ,   -0.5 ],
       [   0.  ,   -0.5 ,    0.5 ,   -0.5 ],
       [   0.  ,    1.5 ,    0.5 ,   -0.5 ]])

"""



"""
A = build_projection_operator(64, n_dir=80)
P = np.array(A.todense())
Wm = build_wavelet_matrix_2d(64)
V = np.dot(P,Wm.T) # coherence: np.max(np.abs(V))
V2 = V.T.dot(V) # Transform Point Spread Function (TPSF)
"""

def tpsf_max(A, B, copy=True):
	"""
	Compute max_{i!=j} |V_{i,j}|
	where V = A B.T, A: probe, B: representation
	Mind the "off-diagonal" constraint
	V: TPSF (square matrix by definition)
	Use copy=False if memory is critical

	Not normalized !
	"""
	V = A.dot(B.T)
	E = np.eye(V.shape[0], M=V.shape[1])
	E[E==1] = np.nan
	if copy:
		V2 = V - E
	else:
		V2 = V
		V2 -= E
	return np.nanmax(np.abs(V2))
	

def coherence(A, B):
	"""
	compute the coherence between "basis" A and B:
		mu(A,B) = sqrt(N)*max_{i,k}( < A_i | B_k > )
	The result is a number between [1, sqrt(N)], where
	1 (resp sqrt(N)) means a low (resp. high) coherence.
	"N" is the dimensionality of the input signals fed to A and B.

	Parameters
	-----------
	A: basis (eg. probes), dimensions: (N_p,N)
		where the input signal is (N, 1)
	B: basis (eg. representation), dimensions: (N_r, N)
	"""
	N = A.shape[1]
	return np.max(np.abs(A.dot(B.T))*sqrt(N))

		


def build_gradient_matrix_2d(n, merged=False):
	inp = np.zeros((n, n))
	res1 = np.zeros((n**2, n**2))
	res2 = np.copy(res1)
	for i in range(n):
		for j in range(n):
			I = i*n+j # "ravel" index
			inp[i,j] = 1
			g1, g2 = gradient(inp)
			res1[:, I] = g1.ravel()
			res2[:, I] = g2.ravel()
			inp[i,j] = 0
	if merged:
		res = np.zeros((2*n**2, n**2))
		res[:n**2, :] = res1
		res[n**2:, :] = res2
		return res
	return (res1, res2)





def norm0(coeffs, eps=1e-6):
	res = 0 
	res += np.sum(np.abs(coeffs[0])>eps)
	for k in range(1, len(coeffs)):
		res += np.sum(np.abs(coeffs[k][0])>eps)
		res += np.sum(np.abs(coeffs[k][1])>eps)
		res += np.sum(np.abs(coeffs[k][2])>eps)
	return res


def dft_matrix_1d(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * pi * 1J / N )
    W = np.power( omega, i * j ) / sqrt(N)
    return W




def lravel2cravel(Nr, Nc):
	"""
	Returns a matrix M such as
	M*x reformats x from a line-ravel format to a column-ravel format.

	Example
	-------
	k = np.arange(12).reshape((3,4))
	print(k)
	print(k.ravel())
	print(lravel2cravel(3,4).dot(k.ravel()))
	"""
	res = np.zeros((Nr*Nc, Nr*Nc))
	L = np.zeros(Nr*Nc)
	L[0] = 1
	for i in range(Nc):
		res[i*Nr, :] = L
		for j in range(1, Nr):
			res[i*Nr+j, :] = np.roll(res[i*Nr+j-1], Nc)
		L = np.roll(L, 1)
	return res
			



def dft_matrix_2d(N):
	"""
	Return the DFT matrix for FFT with 2 dimensions.
	As DFT is separable, let F1 be the DFT matrix for 1D.
	Then DFT(axis = 1), where the input image is ravelled, is:
		F1 0  0 ... 0
		0  F1 0 ... 0
		     ...
		0  ...      F1
	
	Then, DFT(axis=0) (along columns) is:
		F1_1
		F1_2
		...
		F1_N
	Where 
		F1 = f11 ... f1N
		     f21 ... f2N
	
		F1_1 = f11 [N-1 zeros] f12 [N-1 zeros] ... f1N [N-1 zeros]
		       f21 [N-1 zeros] f12 [N-1 zeros] ... f2N [N-1 zeros]
			 ...

		F1_2 = 0 f11 [N-1 zeros] f12 [N-1 zeros] ... f1N [N-2 zeros]
		       0 f21 [N-1 zeros] f12 [N-1 zeros] ... f2N [N-2 zeros]
			 ...
	"""

	W = dft_matrix_1d(N)

	# First, build the "Transform along lines" matrix
	R1 = np.zeros((N**2, N**2), dtype=np.complex128)
	for i in range(N):
		R1[i*N:(i+1)*N, i*N:(i+1)*N] = W
	
	# Then, build the "Transform along columns" matrix
	R2 = np.zeros((N**2, N**2), dtype=np.complex128)
	# Build F1_1 matrix: insert N-1 zeros between entries of R1
	F11 = np.zeros((N, N**2), dtype=np.complex128)
	F11[:, ::N] = W
	R2[:N, :] = F11
	for i in range(1, N):
		R2[i*N:(i+1)*N, :] = np.roll(F11, i, axis=1)
	
	# R1 is "DFT(axis=1)" and R2 is "DFT(axis=0)"
	# However, R2.dot(x) returns a result which is transposed 
	# wrt np.fft.fft(x, axis=0). We have to transpose from a 
	# "line-ravel" format to a "column-ravel" format
	M = lravel2cravel(N,N)
	return M.dot(R2).dot(R1) # *4



def sample_uniform(N, s):
	"""
	build an uniform undersampling matrix.
	"""
	Ns = int(round(N*1./s))
	res = np.zeros((Ns, N))
	res[0, 0] = 1
	for i in range(1,Ns):
		res[i, :] = np.roll(res[i-1, :], s)
	return res
	

def sample_bernoulli(N):
	"""
	build a random sampling with a Bernoulli (coin-toss) distribution.
	"""

	drawn = np.random.randint(2, size=N)
	Ns = drawn.sum()
	res = np.zeros((Ns, N))
	cnt = 0
	for i in range(N):
		if drawn[i]:
			res[cnt, i] = 1
			cnt += 1	
	return res


def build_noiselet(npow):
	"""
	Build a complex noiselets matrix.
	Code adapted from Laurent Duval,
	http://nuit-blanche.blogspot.fr/2008/04/monday-morning-algorithm-15-building.html
	"""
	n = 2**npow
	res = np.zeros((n, 2*n-1), dtype=np.complex128)
	res[:, 0] = 1
	c1 = 1 - 1j
	c2 = 1 + 1j
	L = np.zeros(n//2)
	for i in range(n-1):
		E = []
		E.extend(res[:n:2, i])
		E.extend(L)
		v2x = np.array(E, dtype=np.complex128)
		E = []
		E.extend(L)
		E.extend(res[:n:2, i])
		v2x_1 = np.array(E, dtype=np.complex128)
		res[:, 2*i+1] = c1*v2x + c2*v2x_1
		res[:, 2*i+2] = c2*v2x + c1*v2x_1
	return 1./n * res[:, n-1:]
	








