# 赤道断面纬向流年际比较

> 2026-10-19整理：最终展示入口为[核心结果索引](../../fig/261016_ctrl_selection/README.md)。旧对比图与大型缓存已清理；下文历史输出路径是重算流程示例，不代表文件当前仍保留，实验原始output也须实际存在。完整重建入口为`fig/261016_ctrl_selection/reproduce.sh`。

在项目的 `py312` 环境中，从仓库根目录运行：

```bash
python analysis/comparison/compare_equatorial_u.py --exp 261016_ck_0p07_a4_4e10 --skip-deep-modes
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

带通序列有强自相关，脚本不输出以月数当作独立样本计算的显著性；单实验脚本不计算块自举置信区间，配对检验由 `test_ck_significance.py` 单独完成。36 个月裁剪只是统一处理，并不保证 8 年周期的端点误差完全消失，可分别用 `--trim 24` 和 `--trim 48` 检查结论敏感性，结果会覆盖该实验的指标与图。CMEMS 为再分析，不能视作独立观测真值。对混合参数、边界恢复或风强迫的因果归因需要进一步对比敏感性实验。

## 当前保留的脚本

| 脚本 | 用途 |
| --- | --- |
| compare_equatorial_u.py | 通用模拟—CMEMS赤道断面对齐、缓存与原场比较 |
| compare_a4_full_section.py | 可指定实验列表的全断面、分层与区域原场对比 |
| compare_low_vertical_modes.py | 使用共同 N.mat 基底进行低垂直模态重建与比较 |
| assess_low_mode_reconstruction.py | 同一完整水柱掩膜下检查原场与低模态重建 |
| compare_site_u.py | 142°E、0°站点流速与剖面对比 |
| test_ck_significance.py | 可指定实验与基线的配对分块自举及混合系数提取 |
| plot_ck_a4_grid.py | ck×A4 参数二维图；支持原场和低模态 |
| summarize_joint_ctrl.py | 最终控制实验筛选的汇总，复现流程仍使用 |
| verify_final_ctrl.py | 从原始输出独立核查最终控制实验指标及权重敏感性 |
| assess_western_deep.py | 上述脚本共用的路径、带通、权重及指标函数，同时保留历史评估入口 |
| deep_modes.py | 通用断面脚本可选的深层 CEOF 模块；与物理垂直模态不同 |

旧脚本中的默认实验可能属于历史筛选阶段。新比较应显式指定 `--exp`、`--baseline`、`--out` 等可用参数，先查看各脚本 `--help`。

## 最终结果与复现

正式控制实验为 `261016_ck_0p07_a4_4e10`，主要稳健性配置为 `261008_ck_0p07`。参数、核查及解释边界见[正式记录](../../research/model_control/README.md)，展示结果见[核心结果索引](../../fig/261016_ctrl_selection/README.md)。

从仓库根目录执行：

```bash
bash fig/261016_ctrl_selection/reproduce.sh
```

该流程会重新生成中间断面缓存和过程图，增加磁盘占用；它不会覆盖编号核心展示图。需要原始实验输出及 CMEMS 数据齐全。

`test_ck_significance.py` 可指定实验列表和基线；`--skip-mixing` 仅做自举，`--mixing-only` 仅提取混合系数。置信区间是考虑序列相关的探索性逐项区间，未作多重比较校正。

历史图表使用层中心中点构造的权重，名义“1000–3000 m”实际下界约 2868 m；正式核查另以真实 RF 层界重算。两种定义及共同湿点掩膜的区别见正式记录，新机制分析应明确采用实际层界与积分范围。

## 2026-10-19脚本清理

已移出六个不再参与最终复现的一次性 ck/KPP 混合、谱和阻尼分析脚本，以及过时的优化诊断模板。旧代码与清理前说明备份在[历史目录](../../research/comparison_history/2026-10-19_script_cleanup/README.md)，不作为当前运行入口。模拟输出、核心结果和正式控制实验记录未更改。
