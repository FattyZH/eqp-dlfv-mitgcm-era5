# 当前模式—再分析对比

- [固定1900m输运使用说明](channel_transport.md)：主计算、校验、参数及输出。
- compare_channel_transport.py：ctrl和GLORYS在相同可用湿面积上积分Sv，模式完整至海底另列。
- check_channel_transport.py：解析几何、原始通量、闭合和时间块核验。
- summarize_recent_transport.py：默认2015-12至2024-06观测覆盖期；沿用长记录滤波并标注末端。
- compare_site_theta.py：142E/0N原生深度位温和2.2°C深度诊断。

结果：[输运](../../research/transport/2026-11-13_1900m_ctrl/assessment.md)、[观测时段](../../research/transport/2026-11-13_1900m_ctrl/since_201512/assessment.md)、[位温](../../research/temperature/2026-11-13_142E_equator/assessment.md)。

旧的点位/区域U代理、参数筛选和ctrl源复核代码已[留存](../../archive/retained_2026-11-13/analysis/comparison/)，不等于潜标实测验证代码。正式ctrl配置与选型证据仍见[冻结记录](../../research/model_control/README.md)。
