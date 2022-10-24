from numpy import *
import scipy as sp
from scipy import linalg, sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator
import time

class linear_solver:
    def biCGstab_solver(self, A, b):
        start_time = time.time()
        n=int(A.shape[0])
        print(n)
        A = csc_matrix(A)
        c=1
        A_new = A+ c*sparse.eye(n,n);
        #return A_new
        ilu = splu(A_new)
        #return ilu|
        Mx = lambda x: ilu.solve(x)
        #return Mx
        M = LinearOperator((n, n), Mx)
        print("---Preconditioning took %s seconds ---" % (time.time() - start_time))
        print('Engaging linear solver: biCGstab')
        start_time = time.time()
        phi, exitcode=sparse.linalg.bicgstab(A,b, x0=None, tol=1e-20, maxiter=1000, M=M) #,callback=report)
        if exitcode==0:
            print('Converged! Time for a pint!')
        else:
            print('Not converged, try praying harder next time')
        print("---Linear solver took %s seconds ---" % (time.time() - start_time))
        return(phi)

class report_res:
    def L_norm_res(self, A, phi, b):
        n=int(A.shape[0])
        res=sum(abs(A*(phi)-b))
        L1_norm=sum(abs(A.dot(phi)-b))/n
        L2_norm=sqrt(sum((A.dot(phi)-b)**2)/n)
        print("Final global residual:", res)
        print("Representative residual L_1:", L1_norm)
        print("Representative residual L_2:", L2_norm)
        return(res, L1_norm, L2_norm)