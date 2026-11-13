# 当前分析入口

当前主标准：142E、岛岸至2N、1900m以下、向东为正；主比较期先按2015-12至2024-06，精确首月待原始潜标有效月份核对。现有直接对比对象为GLORYS，再分析不等于潜标观测。

| 目录 | 当前入口 | 用途 |
|---|---|---|
| comparison | [compare_channel_transport.py](comparison/compare_channel_transport.py) | 固定深度Sv输运、同面积ctrl/GLORYS对比 |
| comparison | [check_channel_transport.py](comparison/check_channel_transport.py) | 面积/方向闭合、解析与原始源核查 |
| comparison | [summarize_recent_transport.py](comparison/summarize_recent_transport.py) | 统一观测覆盖期，标注滤波末端 |
| comparison | [compare_site_theta.py](comparison/compare_site_theta.py) | 142E/0N位温及2.2°C深度 |
| mechanism | [compare_ten_day_factorial.py](mechanism/compare_ten_day_factorial.py) | 10天吸收×带阻风四组速度诊断 |
| mechanism | [check_ten_day_factorial.py](mechanism/check_ten_day_factorial.py) | 四组源核验与条件波型质量 |
| common | [说明](common/README.md) | MDS、权重、滤波与条件波型公用函数，不直接执行 |
| quicklook | show_u.py、show_dync.py | 运行输出快速查看 |

完整命令见[输运使用说明](comparison/channel_transport.md)，研究结果入口见[research](../research/README.md)。旧参数筛选、CEOF、早期机制脚本及结果统一在[留存目录](../archive/retained_2026-11-13/README.md)。不要将其旧默认实验当作当前配置。

整理只移动历史分析及其结果，删除可再生Python字节码；原始数据、模型输入/输出和正式ctrl证据保留。
