当前统计口径：按用户指定，平滑相关主结果保留2015-12至2024-06全部103月及首尾反射区域；91月去边界结果仅作敏感性对照。

# 当前分析入口

当前主标准：142E、岛岸至2N、1900m以下、向东为正；主比较期先按2015-12至2024-06，完整共同月份已核对。潜标已接入，主验证使用统一365天平滑；去除边界后的统计期为2016-06至2023-12。

| 目录 | 当前入口 | 用途 |
|---|---|---|
| mechanism | [compare_radius_experiments.py](mechanism/compare_radius_experiments.py) | R99/R98/R97参数核验、共同层厚加权CEOF与观测输运对比 |
| comparison | [audit_bathymetry_failure.py](comparison/audit_bathymetry_failure.py) | AVG/GSM地形连通性、旧pickup及启动失稳审计 |
| comparison | [check_iafilt_sensitivity.py](comparison/check_iafilt_sensitivity.py) | 全103月的周期定义、窗口和边界处理敏感性 |
| comparison | [audit_historical_ctrl.py](comparison/audit_historical_ctrl.py) | 6月3日前Git配置与实际CTRL归档的只读追溯 |
| comparison | [compare_ctrl_ensemble.py](comparison/compare_ctrl_ensemble.py) | 五组初值CTRL输运、统一365天验证与集合离散 |
| comparison | [compare_mooring_transport.py](comparison/compare_mooring_transport.py) | 潜标/CTRL/GLORYS统一365天处理与验证 |
| comparison | [compare_channel_transport.py](comparison/compare_channel_transport.py) | 固定深度Sv输运、同面积ctrl/GLORYS对比 |
| comparison | [check_channel_transport.py](comparison/check_channel_transport.py) | 面积/方向闭合、解析与原始源核查 |
| comparison | [summarize_recent_transport.py](comparison/summarize_recent_transport.py) | 统一观测覆盖期，标注滤波末端 |
| comparison | [compare_site_theta.py](comparison/compare_site_theta.py) | 142E/0N位温及2.2°C深度 |
| mechanism | [compare_tke_floor.py](mechanism/compare_tke_floor.py) | 完成TKE下限/底部组合的输运、分层振幅及混合诊断 |
| mechanism | [diagnose_depth_amplitude.py](mechanism/diagnose_depth_amplitude.py) | CTRL/GL12深度—经度年际振幅及模态叠加诊断 |
| mechanism | [plot_equatorial_agreement.py](mechanism/plot_equatorial_agreement.py) | 赤道断面年际相关与振幅联合图、上层放大及独立相关复核 |
| mechanism | [compare_historical_bandpass.py](mechanism/compare_historical_bandpass.py) | NAS旧260825带通风与当前CTRL/GL12深层振幅及同面积输运比较 |
| mechanism | [plot_historical_bandpass_sections.py](mechanism/plot_historical_bandpass_sections.py) | 三组赤道断面振幅、幅度比和相关图，含上层及深层放大 |
| mechanism | [compute_ctrl_ceof.py](mechanism/compute_ctrl_ceof.py) | 当前CTRL赤道断面CEOF：等格点主结果、RF层厚加权敏感性及重构核查 |
| mechanism | [compare_ctrl_gl12_ceof.py](mechanism/compare_ctrl_gl12_ceof.py) | CTRL/GL12共同湿点、同层厚权重独立CEOF及空间相位对齐比较 |
| mechanism | [diagnose_interannual_frequency.py](mechanism/diagnose_interannual_frequency.py) | CMEMS先2–8年带通的500/1000m以下CEOF、加权局地谱及1900m以下Q谱 |
| mechanism | [check_short_record_spectrum.py](mechanism/check_short_record_spectrum.py) | 九年潜标输运约4.5年峰与CMEMS长短记录、滤波支持长度对照 |
| mechanism | [diagnose_channel_amplitude.py](mechanism/diagnose_channel_amplitude.py) | 相同湿面积142E通道局地振幅与输运抵消 |
| mechanism | [compare_transport_factorial.py](mechanism/compare_transport_factorial.py) | 六组原生断面Q及长期2–8年风×吸收对照 |
| mechanism | [compare_ten_day_factorial.py](mechanism/compare_ten_day_factorial.py) | 10天吸收×带阻风四组速度诊断 |
| mechanism | [check_ten_day_factorial.py](mechanism/check_ten_day_factorial.py) | 四组源核验与条件波型质量 |
| common | [说明](common/README.md) | MDS、权重、滤波与条件波型公用函数，不直接执行 |
| quicklook | show_u.py、show_dync.py | 运行输出快速查看 |

完整命令见[输运使用说明](comparison/channel_transport.md)，研究结果入口见[research](../research/README.md)。旧参数筛选、CEOF、早期机制脚本及结果统一在[留存目录](../archive/retained_2026-11-13/README.md)。不要将其旧默认实验当作当前配置。

整理只移动历史分析及其结果，删除可再生Python字节码；原始数据、模型输入/输出和正式ctrl证据保留。
