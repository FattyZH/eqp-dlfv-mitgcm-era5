# mitkit 检查与精简记录

## 从安装包移除

- `io/parse_file.py`：自制 namelist 解析器，项目内只有 `bin/mitmon` 的
  未使用导入。已删除实现和导出，统一使用现有依赖 `f90nml.read()`。
  Git 历史中原始路径为 `script/mit_utils/parse_file.py`。
- `plotting/` 整体归档：唯一包外调用是配套示例，没有发现生产流程或
  科研分析脚本使用该框架。这不证明其科学功能无用；当前实现缺乏验证，
  因此保留源码和 MAT 资源，但不再作为 `mitkit` 的可用 API。
- `plotting_usage.py`：与框架一起归档，不再作为可运行示例宣传。
- `unused_ocean_utils.py`：保存 `iafilt`、`find_mon_file`。
  未发现项目内调用；前者固定了滤波频段和连续等间隔采样假设，后者
  专用于旧 `run/mnc_exp_*` 布局，并按字符串排序选择目录。

归档文件保留原导入，仅供阅读比较；其中 `mitkit.plotting` 已不存在，
不能直接运行。需要恢复某项功能时，应根据具体分析需求单独迁回并验证。

## 绘图代码的具体问题

- `plotters/ocean_plots.py`：`timeseries_euc` 中 `lat`、`model_label`、
  `ds_ref`、`ref_label` 未定义；`sel(lon_dim=lon)` 使用字面维度名，
  并没有使用 `lon_dim` 变量指定的维度。
- `plotters/oceano_plots.py`：未在包初始化时加载，但示例调用其注册名；
  手动加载又与另一文件重复注册 `timeseries_euc`。
- `plotters/annual_plots.py`：固定 2013–2024 时段、每年 12 个样本、
  经纬度名、142°E 标记和相位换算；还重置全局 Matplotlib 配置、
  替换调用者传入的 Axes，可能通过 `.values` 修改原始数据。
- `utils/common.py`：滤波说明称 FFT/Lanczos，实现却是 Butterworth；
  不应在没有采样频率、缺测和频带验证时当作可靠科研算法。
- `plotter.py`：接口描述与实际注册集合不一致，`list_available` 声明
  返回列表但实际没有 return。完整重建该抽象暂时没有明确需求。

## 保留及修复

- `io/open_mds.py`：被多个分析和预处理脚本使用。保留自动读取起始日期、
  时间步长、GGL90 元数据和平均输出时间校正；修复列表 prefix、
  未注册的原生输出前缀、单条诊断配置和用户 extra_variables 的处理。
  不同时间偏移的多个前缀明确要求分别打开，避免错误共用时间坐标。
- `io/parse_diag.py`：读取诊断统计文本，不是 namelist，f90nml 不能替代。
  修复缺失表头终止标记时的死循环，增加表头与记录格式报错，并修正文档。
- `ocean/utils.py`：保留 notebook 确实调用的 `astar_2d`、
  `largest_connected_region`、`gsmooth2d`、`fillna`，限制星号导出为这四项。
- `ocean/modes.py`：保留有明确科研用途的模态求解器，未因没有本地调用删除。
- `paths.py`：保留，各流程依赖。它是项目路径辅助函数，并非通用海洋算法。

## 尚需科学验证的边界

本次不是科学算法全面认证，以下实现没有悄悄改动：

- `astar_2d` 的曼哈顿启发式没有按最小权重缩放；当权重小于 1
  （包括文档允许的零权重）时，不能保证最优路径；还缺少起终点验证。
- `iafilt` 的气候态去除依赖连续完整采样，不能自动识别缺月；已归档。
- `gsmooth2d` 的整数输入和显式掩膜包含 NaN 的情形需完善。
- `fillna` 对全 NaN 输入没有明确定义的有效填充值。
- `dymodes` 的文档变量 `dz` 与参数 `z` 不符，注释里的 N² 算子与
  实际 1/N² 系数不一致；深度/压力约定、稳定性和模态筛选阈值需用
  解析算例或参考结果核验。此次未改动其数值算法。
- `parse_diag` 的不完整末尾记录、多字段同步及时间重建仍需专项测试；
  本次新增的边界检查不等于覆盖所有运行中写入情况。

目前保留 `>=3.10,<3.13` 作为项目环境范围。此前支持 3.10 下限的
  即时联合类型注解已随绘图模块归档，因此不能再用那一处作为当前
  核心包的硬性下限依据。降低下限需另行验证依赖组合和运行环境。
