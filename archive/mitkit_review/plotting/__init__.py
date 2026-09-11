
from mitkit.plotting.plotter import MitPlotter
from .registry import register_plot, get_plotter, list_plotters
# 加载plots代码文件
from .plotters import (
    annual_plots,
    ocean_plots,
)

__all__ = ["MitPlotter", "register_plot", "get_plotter", "list_plotters"]
