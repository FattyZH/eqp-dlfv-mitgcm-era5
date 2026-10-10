# 赤道断面纬向流年际比较

在项目的 `py312` 环境中，从仓库根目录运行：

```bash
python analysis/comparison/compare_equatorial_u.py --exp 260821_ctrl
python analysis/comparison/compare_equatorial_u.py --exp 260916_ctrl_kpp
python analysis/comparison/compare_equatorial_u.py --exp 260915_ctrl1
```

`--exp` 对应 `output/` 下的实验目录，默认 `260821_ctrl`。使用 `open_mds` 读取 `diag3d/UVEL` 及 `hFacW`，依赖现有 mitkit、xarray、numpy、pandas、scipy 和 matplotlib；绘图使用无界面的 Agg 后端。不会修改现有 CEOF 脚本或模拟输出。

每个实验独立保存到 `fig/<实验名>_cmems_comparison/`。首次运行读取原始数据并缓存断面，后续运行直接使用 `sections.nc`。模拟输出更新后用 `--rebuild-cache` 重建；修改断面选择代码后也需重建。`--skip-deep-modes` 可以跳过深层 CEOF，`--trim 36` 控制带通后的首尾各裁剪月数。

## 对齐与年际信号

- CMEMS：`data/cmems/gl12_uvt_9607-2406_fd.nc` 中的 `uo`。
- 赤道断面：两套数据分别取 0.5°S–0.5°N 湿格点纬向平均。模拟在平均前用 `hFacW > 0` 掩膜；不将陆地的零流速计入平均。
- 时间：按日历月份对齐，忽略月初与诊断平均区间中点的日期差异，只保留共同月份，拒绝缺月或重复月份。输入必须是月平均诊断。
- 经度：130–285°E；深度：CMEMS 覆盖内的模拟层中心（当前约 2.2–2867.6 m）。CMEMS 先做赤道带平均，再线性插值到模拟断面网格，采用两者全时段完整的共同湿格点。
- 两套数据分别去除共同时间段内的逐月气候态，再使用相同四阶 Butterworth 零相位 2–8 年带通。默认首尾各去掉 36 个月：1996-07–2024-06 的输入对应 1999-07–2021-06 的指标期。
- 区域平均先按层厚加权，再对规则经度网格平均。区域平均标准差与逐点标准差的空间平均是不同指标，前者可能因不同深度的流速反相抵消而变小。

## 结果文件

| 文件 | 内容 |
|---|---|
| `section_metrics.png` | CMEMS 平均流、模拟平均偏差、双方年际标准差、逐点相关系数与振幅比 |
| `regional_timeseries.png` | 西、中、东太平洋 × 0–200、200–1000、1000–3000 m 的年际时间序列 |
| `regional_spectra.png` | 去季节但未带通序列的 Welch 谱，独立检查 2–8 年能量；窗口最长 168 个月 |
| `deep_hovmoller.png` | 1000–3000 m 厚度加权流速；经度横轴、时间纵轴，CMEMS、模拟与差值水平排列，共用色标 |
| `deep_ceof.png` | 深层各自 CEOF1 的振幅、空间相位和复时间系数 |
| `regional_metrics.csv` | 九个区域的相关系数、双方标准差和 RMSE |
| `summary.json` | 时间范围、有效格点数和区域指标 |
| `deep_modes.json` | 前三个 CEOF 方差贡献及双方模态的复时间系数相似度矩阵（行：模拟，列：CMEMS） |
| `sections.nc` | 对齐后的原始月平均断面，单位 m/s |
| `interannual_sections.nc` | 裁剪后的 2–8 年流速，单位 m/s |
| `metrics.nc` | 逐点指标，流速和标准差单位 m/s；相关系数及标准差比无量纲 |

CEOF 使用 140–280°E、1000–3000 m 的共同完整湿格点和层厚权重；时间系数均方模为 1，空间模态保留 m/s 单位。模拟 CEOF1 旋转相位，与 CMEMS CEOF1 的时间系数对齐。相位定义为 `arg(spatial mode)`；与现有 `ana_ceof.py` 的负相位符号不同，不能直接混用。为便于对应裁剪后的序列，这里在裁剪后做 Hilbert 变换，其两端仍可能有误差。

## 解读边界

相关性表示同一时期变化的一致程度；振幅比和 RMSE 衡量大小差异。两套各自求得的 CEOF 可能发生模态排序交换或混合，应查看完整相似度矩阵，不能只因都叫 CEOF1 就认定为同一物理模态。深度平均的 Hovmoller 图也不足以单独确定传播速度。

带通序列有强自相关，脚本不输出以月数当作独立样本计算的显著性；目前也未做块自举置信区间。36 个月裁剪只是统一处理，并不保证 8 年周期的端点误差完全消失，可分别用 `--trim 24` 和 `--trim 48` 检查结论敏感性，结果会覆盖该实验的指标与图。CMEMS 为再分析，不能视作独立观测真值。对混合参数、边界恢复或风强迫的因果归因需要进一步对比敏感性实验。

## 混合参数敏感性对比

先为每个实验运行 `compare_equatorial_u.py --exp 实验名`，再运行：

```bash
python analysis/comparison/compare_ck_sensitivity.py --baseline 260915_ctrl1 --experiment 261008_ck_0p07
python analysis/comparison/compare_site_u.py --exp 260821_ctrl 260915_ctrl1 261008_ck_0p07
```

敏感性脚本对齐双方断面缓存并采用共同湿格点，保存西太平均流和年际振幅变化图、分层时间序列与区域指标到 `fig/<experiment>_vs_<baseline>/`。新的0.10完成后将baseline替换为对应实验名。

三组ck实验均完成后，运行 `python analysis/comparison/summarize_ck_suite.py`。默认比较261008_ck_0p07、261008_ck_0p10、261008_ck_0p14，也可通过 `--exp` 指定列表。需先生成各组断面与142°E站点缓存。输出至 `fig/ck_sensitivity_suite/`，包含24/36/48个月裁剪指标和分层Hovmoller。

简单统计检验与实际混合系数检查：`python analysis/comparison/test_ck_significance.py`。固定三组ck实验，在142°E和130–150°E做配对分块自举（3000次、24/36/48个月块长），并检查目标深层月平均混合系数；输出到 `fig/ck_sensitivity_suite/`。95%区间为探索性逐项区间，未作多重比较校正。
