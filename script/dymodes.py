import numpy as np
from scipy.linalg import eig
import seawater as sw

def dymodes(salt,theta,z):
    """
    计算斜压模态函数和模态速度。

    参数:
        theta: 一维位温剖面 (长度为 nz)，从上到下排列。
        salt: 一维盐度剖面 (长度为 nz)，从上到下排列。
        dz: 一维厚度数组 (长度为 nz)，对应每层的厚度。

    返回:
        tuple: 包含以下元素的元组:
            - pmodes (np.ndarray): 斜压模态函数 (nz, n_modes)，未归一化。
            - ce (np.ndarray): 对应的模态速度 (n_modes)。
    """

    # 将输入转换为 numpy 数组，确保数据类型为 float
    theta = np.asarray(theta, dtype=float)
    salt = np.asarray(salt, dtype=float)
    z = np.asarray(z, dtype=float)

    nz = z.size # nz 是水平速度 u,v 所在的层数
    dzw = z[1:] - z[:-1]  # 计算 w 点 (垂向速度点) 之间的厚度, 长度为 nz-1
    dz = np.zeros(z.shape)
    lev = 0
    for i in range(z.size):
        dz[i] = 2*(z[i]-lev)
        lev += dz[i]
    N2 = sw.bfrq(salt, sw.temp(salt, theta, z), z)[0].squeeze()  # 使用 seawater 库计算 N^2

    # --- 构建广义特征值问题的矩阵 A 和 B ---
    # 目标是求解离散化的 Sturm-Liouville 本征方程:
    # d/dz( N^2(z) * dPhi/dz ) = -lambda * Phi
    # 在交错网格和刚盖边界条件下，离散化后形成 A v = lambda B v 的形式
    # A 矩阵代表了左边的微分算子 d/dz(N^2 d/dz)
    # B 矩阵代表了右边的单位算子 I (乘以 Phi)，但在非均匀网格下，它变成了质量矩阵 M。
    A = np.zeros((nz, nz), dtype=float) # 初始化 A 矩阵 (nz x nz)
    B = np.diag(dz) # B 是对角矩阵，对角线元素为 dz (积分权重)

    # --- 离散化微分算子 d/dz(N^2 d/dz) ---
    # 循环遍历每一对相邻的 w 点 (对应 N2[i] 和 dzw[i])
    # 这里使用中心差分近似 d/dz(N^2 dPhi/dz)
    # 在 w 点 i+1/2 处，N^2 dPhi/dz 的值由两侧 u,v 点的 Phi 值决定
    # (N^2 dPhi/dz)|_{w_{i+1/2}} ~ N2[i] * (Phi[i+1] - Phi[i]) / dzw[i]
    # 然后对这个差分再求一次差分 (在 u,v 点 i 处) 得到 d/dz(N^2 dPhi/dz)
    # [N^2 dPhi/dz]_{w_{i+1/2}} - [N^2 dPhi/dz]_{w_{i-1/2}}
    # 这导致 Phi[i-1], Phi[i], Phi[i+1] 的系数
    # 注意边界条件：刚盖边界条件意味着在最上层和最下层 w 点处，dPhi/dz = 0，
    # 因此直接省略边界一侧的一阶差分项，即为离散边界条件。
    for i in range(nz-1):
        A[i,i] += 1.0/(N2[i]*dzw[i])
        A[i,i+1] -= 1.0/(N2[i]*dzw[i])
        A[i+1,i] -= 1.0/(N2[i]*dzw[i])
        A[i+1,i+1] += 1.0/(N2[i]*dzw[i])

    # --- 求解广义特征值问题 ---
    # A v = lambda B v
    # scipy.linalg.eig 可以求解此问题
    # eigvals: 特征值 lambda (对应 1/c^2，c为模态速度)
    # eigvecs: 对应的特征向量 v (即模态函数 Phi 的离散形式)
    eigvals, eigvecs = eig(A, B)

    # 通常特征值和特征向量可能是复数，但对于物理上合理的对称问题，应取实部
    # 但此 A, B 矩阵构造下，结果可能是实数或复数，取决于矩阵的精确对称性
    # 如果特征值是复数，通常意味着矩阵构造有误或物理问题本身不稳定
    eigvals_real, eigvecs_real = eigvals.real, eigvecs.real

    # --- 筛选和排序 ---
    # 物理上，我们只关心正的特征值 (lambda > 0)，因为 c^2 = 1/lambda
    # 设置一个阈值过滤掉接近零或负的特征值（可能来自数值误差或非物理情况）
    pos_mask = eigvals_real > 1e-8
    eigvals_pos = eigvals_real[pos_mask]
    eigvecs_pos = eigvecs_real[:, pos_mask]
    # 按特征值升序排序 (对应模态速度降序排序，一阶模态速度最快)
    sort_idx = np.argsort(eigvals_pos)
    eigvals_sorted = eigvals_pos[sort_idx]
    pmodes = eigvecs_pos[:, sort_idx] # pmodes 的列是按速度排序的模态函数

    # --- 归一化模态函数 ---
    # eig函数返回的特征向量默认是归一化的，一般可注释掉以下归一化步骤
    pmodes *= np.sign(pmodes[0, :])  # 统一模态函数符号，使得最上层为正
    pmodes /= np.sqrt((pmodes**2*dz[:,None]).sum(0)/dz.sum())  # 归一化模态函数, 使得积分 ∫Phi_i^2 dz = z
    # --- 计算模态速度 ---
    # 特征值 lambda = 1 / c^2
    # 因此，模态速度 c = sqrt(1 / lambda)
    ce = 1.0 / np.sqrt(eigvals_sorted)
    return pmodes, ce