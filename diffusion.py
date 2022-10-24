import time
start_time = time.time()
import matplotlib.pyplot as plt

from functions import *
from class_mods.pre_processing import *
from class_mods.solver import *
print('Finished loading functions')

# Meshing
nx=79    #use odd number for better inflated mesh results and even number for uniform mesh
ny=nx    #nx is only coupled to ny for convnience, they can have different cell counts
length=1
r=1.05   #recommended r is 1.05

mesh=meshing()
dx,dy=mesh.inflated(length ,nx , ny ,r) #create inflated mesh
#dx,dy=mesh.uniform(length ,nx ,ny)      #enable for uniform mesh
n=nx*ny
mesh_info=mesh_storage(nx, ny, dx, dy)  #this is a class that stores mesh info for easier function input

# Definie Boundary Conditions
gam=20  #short for gamma (conflict with software)
phi_n=0
phi_e=100
phi_s=0
phi_w=10
h_f=10
phi_ext=300

# Define BC type (options are 'UDF','zeroGradient' and 'fixedValue', note currently the Robin BC is only avaiable for northern face)
north='UDF'  #try 'zeroGradient'
east='fixedValue'
south='zeroGradient'
west='fixedValue'

bc=bc_type()
udf_sp, udf_su=bc.robin(mesh_info, gam, h_f, phi_ext)

# Populate matrix A for boundary cells and b
print('Making matrix A and b')
a_n,a_e,a_w,a_s,A_partial=make_A_inner(nx,ny,dx,dy,gam)

# Function to generate s_p
s_p=make_sp(nx,ny,dx,dy,north,east,south,west,udf_sp,gam)

# Make boundary a_p and s_u
boundary_ap=np.zeros((ny,nx))
boundary_su=np.zeros((ny,nx))
#north
for j in range(ny-1,ny,1):  # ny
        for i in range(0,nx,1):   #nx
            boundary_ap[j,i]=a_e[j,i]+a_s[j,i]+a_w[j,i]-s_p[4,j,i]      #north a_p
            boundary_su[j,i]=gam*dx[j,i]*2*(phi_n)/dx[j,i]              #north s_u
            if north=='UDF':
                boundary_su[j,i]=udf_su[j,i]    #north udf s_u
#east
for j in range(0,ny,1):
        for i in range(nx-1,nx,1):
            boundary_ap[j,i]=a_n[j,i]+a_s[j,i]+a_w[j,i]-s_p[4,j,i]      #east a_p
            boundary_su[j,i]+=gam*dy[j,i]*(2*phi_e)/dx[j,i]             #east s_u
            if east=='UDF':
                boundary_su[j,i]+=udf_su[j,i]   #east s_u
#south
for j in range(0,1,1):
        for i in range(0,nx,1):
            boundary_ap[j,i]=a_n[j,i]+a_e[j,i]+a_w[j,i]-s_p[4,j,i]      #south a_p
            boundary_su[j,i]+=gam*dx[j,i]*(2*phi_s)/dy[j,i]             #south s_u
            if south=='UDF':
                boundary_su[j,i]+=udf_su[j,i]   #south s_u
#west
for j in range(0,ny,1):  # ny
        for i in range(0,1,1):   #nx
            boundary_ap[j,i]=a_n[j,i]+a_e[j,i]+a_s[j,i]-s_p[4,j,i]      #west a_p
            boundary_su[j,i]+=gam*dy[j,i]*(2*phi_w)/dx[j,i]             #west s_u
            if west=='UDF':
                boundary_su[j,i]+=udf_su[j,i]   #north s_u

boundary_ap_l=boundary_ap.reshape(n)
A_boundary=sparse.diags(boundary_ap_l,0)
A=A_partial+A_boundary    #Cobine the two partially constructed matrix A
b=boundary_su.reshape(n)  #linearise matrix b

# linear solver (preconditioner built in)
solver=linear_solver()
phi=solver.biCGstab_solver(A, b)
phi_plot=phi.reshape((ny,nx))

# reporting L1 and L2 residual
residual=report_res()
res, L1_norm, L2_norm=residual.L_norm_res(A, phi, b)

# visualization
print(' Max phi: %s' % (np.max(phi_plot)))
x_cor,y_cor=make_xy_cell_centre(nx,ny,dx,dy)  # makign x, y coordinate for plotting

plt.pcolormesh(x_cor,y_cor,phi_plot,cmap='hot')
plt.colorbar()
plt.contour(x_cor,y_cor,phi_plot,[20,40,60,80,100] ,colors='black')
#plt.scatter(x_cor,y_cor,marker='x')  # enable to see cell centre location
plt.title('$\phi$ Contour, lines represent 20, 40, 60, 80 and 100')
plt.ylabel('$\it{y}$')
plt.xlabel('$\it{x}$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

