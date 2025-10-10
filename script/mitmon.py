import xarray as xr
from ocean_utils import find_mon_file
import matplotlib.pyplot as plt
ts = slice(None, None)
paths = find_mon_file()
key = ['momKE_ave','ETAN_std']
# key.append('RHOAnoma_lv_std')
isbreak = False
with xr.open_mfdataset(paths) as a:
    print(f"目前迭代步数为{a['T'].size}, 时间为{a['T'].values[-1]}")
    for k in key:
        if k not in a:
            raise ValueError(f"变量{k}不在数据集中,可用变量有{list(a.data_vars)}")
        da = a[k].loc[ts].load()
        if da[-1].isnull().item():
            isbreak = True
        da.plot(marker='.',label=k)
if isbreak:
    print("注意: 数据出现null值, 可能已经崩溃!!!")
else:
    print("数据正常, 未出现null值")
plt.gca().set_yscale('log')
plt.legend()
plt.ylabel('')
plt.title(f"nIter:{a['T'].size}")
plt.savefig('监控.jpg', dpi=300)
print("图片已保存为:监控.jpg")