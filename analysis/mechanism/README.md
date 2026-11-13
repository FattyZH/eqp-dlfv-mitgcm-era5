# 当前机制分析

当前入口compare_ten_day_factorial.py、check_ten_day_factorial.py，比较C/W/A10/WA10，保留30天强度敏感性。

[最新评估](../../research/mechanism/2026-11-13_ten_day_factorial/assessment.md)，复现：

```bash
bash research/mechanism/2026-11-13_ten_day_factorial/reproduce.sh
```

这里仍是固定层段速度/低模态诊断，尚未用新1900m通道Sv重算四组。带阻后的波型拟合质量有限，不将东部吸收等同反射关闭。

共同函数已移至analysis/common。最新脚本仍需读取[早期四组缓存](../../archive/retained_2026-11-13/research/mechanism/2026-10-20_four_experiments/)、[吸收审计](../../archive/retained_2026-11-13/research/mechanism/2026-10-20_eastern_absorber_audit/)、[10/30天强度对照](../../archive/retained_2026-11-13/research/mechanism/2026-10-21_absorber_strength/)，均完整留存且已更新路径。
