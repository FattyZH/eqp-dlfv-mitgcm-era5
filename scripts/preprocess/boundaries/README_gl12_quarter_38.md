# GL12 1/4°初始场与OBCS边界场

`gen_bc_4.py`只生成五个初始场与12个南、北、西月气候态边界场。
不读取模型地形或运行配置，不生成候选配置、完整月场、地形或质控图。
另存`manifest.json`记录分组、delR、源路径、文件尺寸、范围与SHA256。

## 脚本头部配置

- `CLIMATOLOGY`：GL12的12个月温盐流速及海面高度气候态。
- `COORDINATES`：提供原生e3t层厚。
- `SOURCE_BATHYMETRY`：GL12原生地形，仅用于识别表层最大四邻域连通海盆。
- `NX, NY`、`GRID_SPACING`、`FIRST_LONGITUDE`、`FIRST_LATITUDE`：目标网格大小、间距和首个格点中心；默认720×192、0.25°、112°E/23.5°S。
- `LAYER_EDGES`：原生零起始分组边界，相邻a,b合并原生层`[a:b]`。
  Nr自动为边界数减1，层厚按e3t求和，场按e3t加权平均。
  从0开始、严格递增，末边界不能超过原生层数；末边界之后的层被舍弃。
- `INITIAL_MONTHS`：五个初始场先平均这些月份，默认`[12,1]`。
- `SMOOTH_SIGMA`、`SMOOTH_THRESHOLD`、`SOURCE_HALO_DEGREES`：水平平滑和源区域外扩参数。
- `OUTPUT_DIR`：独立输出目录；None自动取`data/preprocessed/gl12_quarter_<Nr>`。
- `OVERWRITE`：默认True，允许重复运行并覆盖同名初始场、边界场及manifest；
  不删除其他文件。设为False时要求输出目录不存在或为空。
- `DRY_RUN`：True只核查输入和网格；False生成数据。

```python
LAYER_EDGES = [0, 5, 9, 12, 14] + list(range(16, 50))  # 38层，舍弃原生第49层
# LAYER_EDGES = [0, 4, 7, 10, 12] + list(range(14, 50))  # 40层
# LAYER_EDGES = list(range(51))                         # 原生50层
```

所有参数直接修改脚本头部，没有命令行解析。

## 空间处理与输出

先在源地形上保留最大的表层连通海盆，不加人工封闭线。逐变量、逐月、逐原生深度
做归一化二维高斯平滑（默认sigma=1、有效权重>0.1、reflect），采样至1/4°，
随后用同一水平层的最近有效值填补缺测，再垂向合并。被剔除海盆不能提供填补值。
不跨时间/深度填补；整层无供体时报错。数值陆点也填有限占位值，实际湿点由模式另行控制。
U/V延续旧脚本的同坐标采样，不额外做C网格半格点移动。

- `init/uo_4_<Nr>.bin`、vo、thetao、so：形状(Nr,NY,NX)。
- `init/zos_4_<Nr>.bin`：形状(NY,NX)。
- `obcs/OBSu_4_<Nr>.bin`、OBSv、OBSt、OBSs，以及对应北侧OBN、西侧OBW：
  南北形状(12,Nr,NX)，西侧(12,Nr,NY)，按1月至12月排序。
  每月处理完整目标场后，取首行、末行、首列作为南、北、西边界；完整月场不写磁盘。
- 全部big-endian float32、C顺序，水平索引最快变化。

## 运行与检查

在仓库根目录运行（先按需编辑配置）：

```bash
OPENBLAS_NUM_THREADS=1 /public/home/zhanghang/.conda/envs/py312/bin/python \
scripts/preprocess/boundaries/gen_bc_4.py
```

测试：

```bash
OPENBLAS_NUM_THREADS=1 /public/home/zhanghang/.conda/envs/py312/bin/python \
-m unittest discover -s tests/preprocess -v
```

包含可配置层厚积分守恒、连通性、缺测填补、二进制排列，以及无需模型文件即可生成
17个初始/边界二进制的端到端测试。每个输出生成时检查有限值、尺寸与二进制往返一致性。
模式Nr、delR（见manifest）、边界位置及文件路径需自行匹配配置。
旧版已生成目录保留不动；其中旧候选配置和完整月场不代表当前脚本还会生成这些文件。

## 性能

边界仍从逐月完整目标场提取，以保留二维近邻供体选择口径。
同一变量、原生深度层在不同月份的有效掩膜一致时，复用平滑分母及近邻索引；
逐次核对掩膜，变化时自动重算。缓存仅驻内存，处理下一个变量时释放。
580×2180合成源网格、8次重复处理测试中，核心函数耗时2.252秒降至0.427秒，
输出逐值相同；此约5.28倍为核心函数测试，不代表含读取与合并的整脚本加速比。
