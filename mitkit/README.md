# mitkit

MITgcm 实验使用的辅助 Python 库。此目录是独立的包项目根目录：

```text
pyproject.toml   包构建、Python 版本和依赖配置
src/mitkit/     可导入源码
tests/         库的回归测试
```

从实验仓库根目录安装：

```bash
python -m pip install --no-deps --no-build-isolation -e ./mitkit
python -m unittest discover -s mitkit/tests
```

导入方式不变：`from mitkit.io import open_mds`。
`project_root()` 优先采用 `WORK_DIR`；未设置时，从源码位置向上寻找
同时包含 `code/` 和 `input/` 的实验目录。独立安装到其他位置时应设置
`WORK_DIR`。本配置不管理 MITgcm 的编译、运行和数据目录。
