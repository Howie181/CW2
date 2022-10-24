from numpy import *

class meshing:
    def uniform(self, length, nx, ny):  # make uniform mesh
        dx=zeros((ny,nx))
        dy=zeros((ny,nx))
        dx[:]=length/nx
        dy[:]=length/ny
        return dx, dy
    def inflated(self, length, nx, ny, r):
        dx=zeros((ny,nx))
        dy=zeros((ny,nx))
        dx_first=((1-r)/(1-r**(nx/2)))*(length)/2
        dx[:,0]=dx_first
        print("First cell is %s m wide" % (dx_first))
        dy_first=((1-r)/(1-r**(ny/2)))*(length)/2
        dy[0,:]=dy_first
        for j in range(0,ny,1):
            for i in range(1,nx,1):
                if sum(dx[j,:])<=length/2:
                    dx[j,i]=dx[j,i-1]*r
                else:
                    dx[j,i]=dx[j,i-1]/r
        for j in range(1,ny,1):
            for i in range(0,nx,1):
                if sum(dy[:,i])<=length/2:
                    dy[j,i]=dy[j-1,i]*r
                else:
                    dy[j,i]=dy[j-1,i]/r
        return dx,dy

class mesh_storage:
    def __init__(self, nx, ny, dx, dy):
        self.a = nx
        self.b = ny
        self.c = dx
        self.d = dy

class bc_type:
    def robin(self, mesh_info, gam, h_f, phi_ext):  # make uniform mesh
        nx=mesh_info.a
        ny=mesh_info.b
        dx=mesh_info.c
        dy=mesh_info.d
        udf_sp=-2*gam*(dx[ny-1,:]/dy[ny-1,:])*(1-1/(1+h_f*dy[ny-1,:]/(2*gam)))
        udf_su=zeros((ny,nx))
        for j in range(ny-1,ny,1):  # ny
            for i in range(0,nx,1):   #nx
                udf_su[j,i]=2*gam*dx[j,i]*phi_ext/(dy[j,i]+2*gam/h_f)
        return udf_sp, udf_su