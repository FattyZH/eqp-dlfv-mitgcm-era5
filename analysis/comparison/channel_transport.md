# 固定深度通道输运

主脚本compare_channel_transport.py，默认计算ctrl与现有GLORYS的142E、岛岸至2N、1900m以下Sv输运；向东为正。允许用户确认的44层资料，自动报告可用/缺失面积，主比较采用共同湿面积；模式完整至海底另列。并非旧的点位或区域平均速度代理。

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/eqp-mpl \
/public/home/zhanghang/.conda/envs/py312/bin/python analysis/comparison/compare_channel_transport.py
```

常用参数：--exp EXP、--upper-depth 1900、--north 2、--reference FILE、--coordinates FILE、--bathymetry FILE、--output DIR、--workers 16。--require-full-depth用于严格要求再分析全海底覆盖，否则在几何检查后停止。默认结果目录按实验与深度命名，避免新实验覆盖ctrl。

静态GLORYS文件默认在/public/home/zhanghang/data。模式142E位于两个U面之间，逐面原生通量计算后插值；使用hFacW/DYG/RF、部分层和北界裁剪。共同面积由稀疏交叠算子生成，不能把缺失深度填0或把两个原生地形的输运误称同面积。

源缓存签名覆盖输入文件路径、尺寸/mtime、静态网格及算法版本。修改提取算法时提高ALGORITHM或移走source_sections.nc/source_signature.json。check_channel_transport.py为默认ctrl结果执行解析几何/单位、源数据和闭合核验，并给出配对时间块不确定性。

输出：transports.csv、filtered_transports.csv、comparison_metrics.csv、upper_depth_sensitivity.csv、source_sections.nc、geometry_and_latitude_transport.nc、geometry.json、summary.json、validation.json。数据与报告在research/transport/2026-11-13_1900m_ctrl，图在fig/261113_transport_261016_ctrl。完整定义与当前结论见对应assessment.md。

当前正式观测期先按2015-12至2024-06（103月）统一，运行summarize_recent_transport.py（默认该首月）或--start指定首月。长记录仅保留作滤波支撑和背景；窗口r含末端时须明确标注。观测原始序列尚未读取，精确首月以后续有效观测月为准。
