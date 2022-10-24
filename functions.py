## Functions are stores here

import time
from scipy.sparse import csc_matrix
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator

def make_sp(nx, ny, dx, dy, north, east, south, west, udf, gam):
    # Calculate local s_p
    s_p = np.zeros((5, ny, nx))
    if north != 'zeroGradient':  # north face
        s_p[0, ny - 1, :] = -2 * gam * dx[ny - 1, :] / dy[ny - 1, :]
    if north == 'UDF':
        s_p[0, ny - 1, :] = udf

    if east != 'zeroGradient':   # east face
        s_p[1, :, nx - 1] = -2 * gam * dy[:, nx - 1] / dx[:, nx - 1]
    if east == 'UDF':
        s_p[1, :, nx - 1] = udf

    if south != 'zeroGradient':  # south face
        s_p[2, 0, :] = -2 * gam * dx[0, :] / dy[0, :]
    if south == 'UDF':
        s_p[2, 0, :] = udf

    if west != 'zeroGradient':   # west face
        s_p[3, :, 0] = -2 * gam * dy[:, 0] / dx[:, 0]
    if west == 'UDF':
        s_p[3, :, 0] = udf

    s_p[4] = s_p[0] + s_p[1] + s_p[2] + s_p[3]  # compute s_p matrix
    return (s_p)


def make_A_inner(nx, ny, dx, dy, gam):
    a_n = np.zeros((ny, nx))
    a_e = np.zeros((ny, nx))
    a_s = np.zeros((ny, nx))
    a_w = np.zeros((ny, nx))
    n = ny * nx
    # a_n
    for j in range(0, ny - 1, 1):  # ny
        for i in range(0, nx, 1):  # nx
            a_n[j, i] = gam * dx[j, i] / (dy[j, i] / 2 + dy[j + 1, i] / 2)
    # a_e
    for j in range(0, ny, 1):
        for i in range(0, nx - 1, 1):
            a_e[j, i] = gam * dy[j, i] / (dx[j, i] / 2 + dx[j, i + 1] / 2)
            # a_s
    for j in range(1, ny, 1):
        for i in range(0, nx, 1):
            a_s[j, i] = gam * dx[j, i] / (dy[j, i] / 2 + dy[j - 1, i] / 2)
            # a_w
    for j in range(0, ny, 1):
        for i in range(1, nx, 1):
            a_w[j, i] = gam * dy[j, i] / (dx[j, i] / 2 + dx[j, i - 1] / 2)
    # Internal cells
    a_p = np.zeros((ny, nx))
    for j in range(1, ny - 1, 1):
        for i in range(1, nx - 1, 1):
            a_p[j, i] = a_n[j, i] + a_e[j, i] + a_s[j, i] + a_w[j, i]

    ac_linear = np.zeros((4, n))
    ac = np.zeros((4, ny, nx))
    ac[0] = a_n
    ac[1] = a_e
    ac[2] = a_s
    ac[3] = a_w
    for k in range(0, 4, 1):
        ac_linear[k, :] = ac[k, :, :].reshape(n)

    # linearise matrix for sparse matrix
    a_n_l = -ac_linear[0, 0:n - nx]
    a_e_l = -ac_linear[1, 0:n - 1]
    a_s_l = -ac_linear[2, nx:n]
    a_w_l = -ac_linear[3, 1:n]
    a_p_l = a_p.reshape(n)

    A = sparse.diags(a_s_l, -nx) + sparse.diags(a_w_l, -1) + sparse.diags(a_p_l, 0) + sparse.diags(a_e_l,
                                                                                                   +1) + sparse.diags(
        a_n_l, +nx)
    return (a_n, a_e, a_w, a_s, A)


def make_xy_cell_centre(nx, ny, dx, dy):
    x_cor = np.zeros((ny, nx))
    y_cor = np.zeros((ny, nx))

    x_cor[:, 0] = dx[0, 0] / 2
    y_cor[0, :] = dy[0, 0] / 2

    for i in range(1, nx, 1):
        x_cor[:, i] = x_cor[:, i - 1] + dx[0, i - 1] / 2 + dx[0, i] / 2
    for j in range(1, ny, 1):
        y_cor[j, :] = y_cor[j - 1, :] + dy[j - 1, 0] / 2 + dy[j, 0] / 2
    return (x_cor, y_cor)