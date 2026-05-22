from mit_utils import parse_file ,parse_diag
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def monitor_simulation(ind=-1,keys=None,out_path=None,fig_name='监控'):
    """
    监控 MITgcm 诊断输出，绘制指定变量的时间序列（对数纵轴），并检查是否出现 NaN。
    
    Parameters:
    - output_file (str): 输出图片文件名，默认 '监控.jpg'
    - keys (list of str): 要绘制的变量名列表，默认 ['momKE_ave', 'ETAN_std']
    """
    if out_path is None:
        out_path = Path.home() / 'eqp-dlfv-mitgcm-era5/output'

    
    # 获取当前目录下的所有文件夹并排序（虽然未使用，但保留原逻辑）
    folders = [f for f in out_path.iterdir() if f.is_dir()]
    folders.sort()
    folder = folders[ind] if folders else '.'
    diag = parse_file(folder / 'data.diagnostics')

    stat_name = diag['DIAG_STATIS_PARMS']['stat_fName(1)']
    cal = parse_file(folder / 'data.cal')
    d1, d2 = cal['CAL_NML']['startDate_1'], cal['CAL_NML'].get('startDate_2', 0)
    s = f"{d1:08d}{d2:06d}"
    ref_date = f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"

    # 加载诊断数据
    a = parse_diag(folder/f'{stat_name}.0000000000.txt',ref_date=ref_date)
    
    # 获取最后迭代步和时间
    niter = int(a['iter'][-1])
    ntime = a['time'][-1]
    compelete = a['complete']
    print(f"目前迭代步数为{niter}, 时间为{ntime}, 结束标志为{compelete}。")

    # 检查变量是否存在
    available_vars = list(a['fields'])
    for k in keys:
        if k.split('_')[0] not in a:
            raise ValueError(f"变量 {k} 不在数据集中。可用变量有: {available_vars}")

    # 初始化标志
    isbreak = False

    # 绘图
    plt.figure(figsize=(8, 5))
    for k in keys:
        key, col = k.split('_')[:2]
        if col in a['cols']:
            col_idx = a['cols'].index(col)
            da = a[key][0,:,col_idx]  # ts = 全部时间
        else:
            raise ValueError(f"列 {col} 不在数据集中。可用列有: {a['cols']}")
        # 检查最后一个值是否为 NaN
        if np.isnan(da[-1]):
            isbreak = True
        plt.plot(a['time'],da,marker='.', label=k)

    if isbreak:
        print("注意: 数据出现 null 值，可能已经崩溃!!!")
    else:
        print("数据正常，未出现 null 值")

    # 设置图形样式
    plt.gca().set_yscale('log')
    plt.legend()
    plt.ylabel('')
    plt.title(f"nIter: {niter}")
    plt.tight_layout()
    save_path = out_path/f"{fig_name}_{folder.name}.jpg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存为: {save_path}")


def main():
    # 可自定义要监控的变量
    key_list = ['momKE_ave', 'ETAN_std']
    for i in range(-1,0):
        monitor_simulation(ind=i, keys=key_list)


if __name__ == "__main__":
    main()