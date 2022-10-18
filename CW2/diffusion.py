import time
start_time = time.time()

import inspect
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg, sparse
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy import stats
from scipy.sparse import csc_matrix

from numpy import *
import numpy as np
from math import *
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import eye
from scipy.sparse import linalg
import scipy.sparse.linalg
from functions import *
print('Finished loading functions')

# Meshing
nx=9  #odd number strongly preferred, (nicer mesh ^^)
ny=nx
length=1
r=1.2  #mesh inflation growth ratio (Ansys fluent default 1.2)

dx,dy=inflated_mesh(nx,ny,r,length)  #make inflated mesh
#dx[:]=length/nx #enable if uniform mesh desired
#dy[:]=length/ny
n=nx*ny

# Define BC
gam=20  #short for gamma (conflict with software)
phi_n=0
phi_e=100
phi_s=0
phi_w=10
h_f=10
phi_ext=300

north='UDF'
east='fixedValue'
south='zeroGradient'
west='fixedValue'

# Define udf if desired
udf_sp = -dx[ny - 1, :] * h_f * (2 * gam / dy[ny - 1, :]) / (h_f - (2 * gam / dy[ny - 1, :]))
s_p = make_sp(nx, ny, dx, dy, north, east, south, west, udf_sp, gam)

udf_su = np.zeros((ny, nx))
for j in range(ny - 1, ny, 1):  # ny
    for i in range(0, nx, 1):  # nx
        udf_su[j, i] = dx[j, i] * h_f * phi_ext * (1 - h_f / (h_f - (2 * gam / dy[j, i])))

        # udf_sp=udf_su=0    #Enable if no udf is applied

# Populate matrix A
a_n, a_e, a_w, a_s, A_partial = make_A_inner(nx, ny, dx, dy, gam)

# Make boundary a_p and s_u
boundary_ap=np.zeros((ny,nx))
boundary_su=np.zeros((ny,nx))
#north
for j in range(ny-1,ny,1):  # ny
        for i in range(0,nx,1):   #nx
            boundary_ap[j,i]=a_e[j,i]+a_s[j,i]+a_w[j,i]-s_p[4,j,i]      #north a_p
            if north=='UDF':
                boundary_su[j,i]=udf_su[j,i] #dx[j,i]*h_f*phi_ext*(1-h_f/(h_f-(2*gam/dy[j,i])))   #north s_u
#east
for j in range(0,ny,1):  # ny
        for i in range(nx-1,nx,1):   #nx
            boundary_ap[j,i]=a_n[j,i]+a_s[j,i]+a_w[j,i]-s_p[4,j,i]
            boundary_su[j,i]+=gam*dy[j,i]*(2*phi_e)/dx[j,i]
#south
for j in range(0,1,1):  # ny
        for i in range(0,nx,1):   #nx
            boundary_ap[j,i]=a_n[j,i]+a_e[j,i]+a_w[j,i]-s_p[4,j,i] #south
            boundary_su[j,i]+=gam*dx[j,i]*(2*phi_s)/dy[j,i]
#west
for j in range(0,ny,1):  # ny
        for i in range(0,1,1):   #nx
            boundary_ap[j,i]=a_n[j,i]+a_e[j,i]+a_s[j,i]-s_p[4,j,i]
            boundary_su[j,i]+=gam*dy[j,i]*(2*phi_w)/dx[j,i]

boundary_ap_l=boundary_ap.reshape(n)
A_boundary=sparse.diags(boundary_ap_l,0)
A=A_partial+A_boundary
b=boundary_su.reshape(n,1)
#print(boundary_su)

M=preconditioner(A,n,1)
print("---Constructing Matrix A and b took %s seconds ---" % (time.time() - start_time))

print('Engaging biCGstab')
start_time = time.time()
y, exitcode=sparse.linalg.bicgstab(A,b, x0=None, tol=1e-12, maxiter=50000, M=M)
print(exitcode)
print("---Linear solver took %s seconds ---" % (time.time() - start_time))
y_plot=y.reshape((ny,nx)) #restructuring data

x_cor,y_cor=make_xy_cell_centre(nx,ny,dx,dy)

xv, yv = np.meshgrid(x_cor[0,:], y_cor[0,:], indexing='ij')
plt.pcolormesh(x_cor,y_cor,y_plot,cmap='hot') #,vmin=0, vmax=100)
plt.colorbar()
plt.scatter(x_cor,y_cor,marker='x')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()