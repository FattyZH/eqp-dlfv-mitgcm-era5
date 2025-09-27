import numpy as np
import heapq
from scipy.ndimage import label, gaussian_filter, distance_transform_edt

def astar_2d(weight_matrix, start, end):
    """
    使用A*算法在二维网格中寻找最短路径

    参数：
        weight_matrix (np.ndarray): 二维权重矩阵,每个点的权重值为0表示可通过,inf或负值表示障碍
        start (tuple): 起点坐标 (row, col)
        end (tuple): 终点坐标 (row, col)

    返回：
        tuple: (path, total_weight)
               path 为从起点到终点的路径坐标列表;若无可行路径,返回 (None, np.inf)
    """
    rows, cols = weight_matrix.shape
    visited = np.zeros_like(weight_matrix, dtype=bool)
    dist = np.full_like(weight_matrix, np.inf, dtype=float)
    prev = np.full((rows, cols, 2), -1, dtype=int)
    heap = []
    def heuristic(a, b):
        # 使用曼哈顿距离作为启发式估计
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    dist[start] = weight_matrix[start]
    heapq.heappush(heap, (dist[start]+heuristic(start,end), dist[start], start))
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    while heap:
        _, cur_dist, (r, c) = heapq.heappop(heap)
        if visited[r, c]: 
            continue
        visited[r, c] = True
        if (r, c) == end:
            # 找到终点后退出循环
            break
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and not visited[nr, nc]:
                w = weight_matrix[nr, nc]
                if w == np.inf or w < 0: # 遇到障碍,跳过该邻居
                    continue
                new_dist = cur_dist + w
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    prev[nr, nc] = [r, c]
                    heapq.heappush(heap, (new_dist+heuristic((nr,nc),end), new_dist, (nr,nc)))
    # 回溯构建路径
    path = []
    cur = end
    if dist[end] == np.inf:
        return None, np.inf
    while cur != start:
        path.append(cur)
        cur = tuple(prev[cur])
    path.append(start)
    path.reverse()
    return path, dist[end]

def largest_connected_region(binary_matrix):
    """
    获取二值矩阵中最大的连通区域

    参数：
        binary_matrix (np.ndarray): 布尔矩阵,其中True代表目标区域

    返回：
        np.ndarray: 布尔矩阵,标识出最大连通区域位置;若不存在连通区域,则返回全False矩阵
    """
    # 对所有区域进行标记
    labeled_array, num_features = label(binary_matrix)
    if num_features == 0:
        # 无任何连通区域,返回全False数组
        return np.zeros_like(binary_matrix, dtype=bool)
    # 统计各区域的元素个数
    region_sizes = np.bincount(labeled_array.ravel())
    region_sizes[0] = 0  # 排除背景（标签为0）
    # 确定最大区域的标签
    largest_label = region_sizes.argmax()
    return labeled_array == largest_label

def gsmooth2d(data, sigma=1.0, mask=None, norm=True, threshold=0.5, fill_value=np.nan):
    """
    对二维数据场进行高斯平滑,可选择掩膜及归一化

    参数：
        topo (np.ndarray): 原始地形数组,可能包含NaN值
        sigma (float): 高斯滤波的标准差
        mask (np.ndarray, optional): 掩膜数组,指示有效数据位置;默认为根据finite值生成的数组
        norm (bool): 是否对滤波结果进行归一化处理
        threshold (float): 阈值,用于判断掩膜模糊后的有效区域,默认值为0.5


    返回：
        np.ndarray: 平滑后的地形数据,掩膜无效区域保留为NaN
    """
    if mask is None:
        mask = np.isfinite(data)
    # 用0填充NaN值,确保数据连续性
    data_filled = np.where(mask, data, 0)
    sigma_tuple = (0,) * (data.ndim - 2) + (sigma,)*2 if data.ndim > 2 else sigma
    data_blurred = gaussian_filter(data_filled, sigma=sigma_tuple)
    sigma_tuple = (0,) * (mask.ndim - 2) + (sigma,)*2 if mask.ndim > 2 else sigma
    mask_blurred = gaussian_filter(mask.astype(float), sigma=sigma_tuple)
    if norm:
        with np.errstate(invalid='ignore', divide='ignore'):
            data_blurred = data_blurred / mask_blurred
    # 使用阈值检查恢复原图区域
    return np.where(mask_blurred > threshold, data_blurred, fill_value)

def fillna(arr):
    """
    利用距离变换填充数组中的NaN值

    参数：
        arr (np.ndarray): 可能包含NaN值的输入数组

    返回：
        np.ndarray: 用最近的有效值替换NaN后的数组
    """
    filled = arr.copy()
    if np.isnan(arr).any():
        mask = np.isnan(arr)
        # 计算每个NaN位置最近非NaN值的位置索引
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        filled[:] = arr[tuple(idx)]
    return filled