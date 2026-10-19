"""Compare original and low-mode fields on exactly the same common mask."""
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from compare_low_vertical_modes import ROOT,bandpass,weights,metrics
OUT=ROOT/'fig/low_vertical_modes_comparison'

def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='fig/low_vertical_modes_comparison')
    OUT=ROOT/p.parse_args().out
    ref=xr.load_dataset(OUT/'CMEMS_reconstruction.nc');rows=[]
    info=json.loads((OUT/'summary.json').read_text())
    for exp in info['experiments']:
        ds=xr.load_dataset(OUT/f'{exp}_reconstruction.nc')
        for west,east in [(130,150),(140,170),(170,220),(220,280)]:
            for lo,hi in [(0,300),(300,1000),(1000,3000),(1000,1500),(1500,2000),(2000,3000)]:
                def avg(a):return a.sel(lon=slice(west,east)).weighted(weights(a,lo,hi)).mean(('depth','lon'))
                for key in ['original_u','low_u']:
                    m,c=avg(ds[key]),avg(ref[key])
                    rows.append(dict(experiment=exp,representation=key,longitude=f'{west}-{east}',depth=f'{lo}-{hi}',mean_bias_cm_s=float((m-c).mean()*100),**metrics(bandpass(m,36).values,bandpass(c,36).values)))
    pd.DataFrame(rows).to_csv(OUT/'original_vs_low_metrics.csv',index=False)
    print('Same-mask original versus low-mode comparison saved')

if __name__=='__main__':main()
