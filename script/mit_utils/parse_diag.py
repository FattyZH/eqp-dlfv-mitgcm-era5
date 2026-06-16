import numpy as np
import re
_COL_NAME = ["ave", "std", "min", "max", "vol"]
_RE_FIELD = re.compile(
    r"field\s*:\s*(\w+)\s*;\s*Iter\s*=\s*(\d+)\s*;\s*region\s*#\s*(\d+)\s*;\s*nb\.Lev\s*=\s*(\d+)"
)
_FIX_EXP = re.compile(r'(?<=\d)([+-])(?=\d)')

def parse_diag(filename,ref_date=None):
    """
    解析 stat 文件，返回 dict：
      - 表头键值（frequency, phase, n_regions, fields, n_levels）
      - data[field_name]: ndarray, shape (n_times, n_levels, 6)
        列顺序: k, avg, std, min, max, vol
      - complete: bool，文件是否包含正常结束标志
    """
    result = {'cols':_COL_NAME}
    data = {}
    data_lv = {}
    Iters = []
    complete = False
    
    with open(filename) as f:

        # 表头
        while True:
            line = f.readline()
            if "end of header" in line:
                break
            if line.startswith("#") and ":" in line:
                key, _, val = line.lstrip("# ").partition(":")
                key, val = key.strip().lower(), val.strip()
                if   "frequency" in key: result["frequency"] = abs(float(val))
                elif "phase"     in key: result["phase"]     = float(val)
                elif "regions"   in key: result["n_regions"] = int(val)
                elif "fields"    in key: result["fields"]    = val.split()
                elif "nb of lev" in key: result["n_levels"]  = list(map(int, val.split()))
        for i, field in enumerate(result["fields"]):
            data[field] = [[] for i in range(result["n_regions"] + 1)]
            if result["n_levels"][i] > 1:
                data_lv[field+'_lv'] = [[] for i in range(result["n_regions"] + 1)]
        # 数据体
        while (line := f.readline()):
            line = line.strip()
            # 注释及输出结束标志
            if line.startswith("#"):
                if "records End here" in line:
                    complete = True
                    break
                else:
                    continue
            if not line:
                continue
            # 字段标题
            m = _RE_FIELD.search(line)
            field,Iter,region,nlev = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if not Iters or Iter != Iters[-1]: 
                Iters.append(Iter)
            nlev = 0 if nlev == 1 else nlev
            buf = np.full((nlev+1, len(_COL_NAME)), np.nan)
            f.readline()  # 跳过列标题
            for i in range(nlev+1):
                line = f.readline()
                if not line:
                    break
                buf[i] = list(map(float, _FIX_EXP.sub(r'E\1', line).split()[1:]))
            data[field][region].append(buf[0])  # 全层
            if nlev > 1:
                data_lv[field+'_lv'][region].append(buf[1:])  # 分层
        
        result["complete"] = complete
        result['iter'] = np.array(Iters)
        if ref_date:

            ref_date = np.array(ref_date,dtype='M8[s]')
        else:
            ref_date = np.datetime64('0000-01-01T00:00:00')
        time = (np.arange(len(Iters))+1) * result['frequency'] + result['phase']
        result['time'] = ref_date + time.astype('i8')
        result |= {key: np.array(value) for key,value in data.items()}
        result |= {key: np.array(value) for key,value in data_lv.items()}
    return result