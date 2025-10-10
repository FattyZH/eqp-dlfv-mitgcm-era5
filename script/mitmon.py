import xarray as xr
from ocean_utils import find_mon_file
import matplotlib.pyplot as plt
ts = slice(None, None)
paths = find_mon_file()
key = ['momKE_ave','ETAN_std']
# key.append('RHOAnoma_lv_std')

with xr.open_mfdataset(paths) as a:
    print(f"目前迭代步数为{a['T'].size}, 时间为{a['T'].values[-1]}")
    for k in key:
        if k not in a:
            raise ValueError(f"变量{k}不在数据集中,可用变量有{list(a.data_vars)}")
        a[k].loc[ts].plot(marker='.',label=k)

plt.gca().set_yscale('log')
plt.legend()
plt.ylabel('')
plt.title(f"nIter:{a['T'].size}")
plt.savefig('监控.jpg')
print("图片已保存为:监控.jpg")