# %%
import sys
sys.path.append ('../')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio
import xarray as xr
latlim = slice(-5.5,5.5)
import mit_utils
def hmnc(z, p):
    n = z.shape[0]
    e1 = np.exp(-2j*np.pi/p*np.arange(n))
    e2 = np.exp(2j*np.pi/p*np.arange(n))
    mf1 = np.tensordot(z, e1, (0, 0))/n
    mf2 = np.tensordot(z, e2, (0, 0))/n
    r = (abs(mf1)**2+abs(mf2)**2)/np.var(z, 0)
    major = abs(mf1)+abs(mf2)
    minor = abs(mf1)-abs(mf2)
    inc = np.angle(mf1*mf2)/2
    pha = inc-np.angle(mf1)
    return r, pha, major, minor, inc
esdata = mit_utils.open_mds('../../output/251222_122739',prefix='dync')
lon1 = esdata['XG'].values
lat1 = esdata['YC'].values
dep1 = -esdata['Z'].values
u1 = esdata['UVEL'][-180:, :, lat1 == 0, :].squeeze().values #取最后15年赤道断面数据
u1[:,np.all(u1 == 0, axis=0)] = np.nan
r1, ph1,maj1 = hmnc(u1, 12)[:3]
mph1 = (12/(2*np.pi)*ph1+.5) % 12+1
tray = sio.loadmat('rossbytray.mat', squeeze_me=True)
tray['x'] = tray['x']-2
# %%

plt.rcdefaults()
plt.rcParams.update(
    {
        'axes.facecolor': 'gray',
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'STIXGeneral',
        'mathtext.it': 'STIXGeneral:italic',
        'mathtext.bf': 'STIXGeneral:bold',
        'mathtext.sf': 'STIXGeneral:bold:italic',
        'image.cmap': 'RdYlBu_r',
        'contour.negative_linestyle': '-',
        'contour.linewidth': 0.8,
        'font.size': 15.0,
    })
mor = ([142, 142, 141.4,140], [-1, 0, -1.7,2])
txdt = {'fontsize': 14, }
tklbdt = {'fontsize': 14, }
btxdt = {'fontsize': 16, 'weight': 'bold'}
cbdtw = dict(orientation='horizontal', aspect=25,pad=0.2)
cbdt = dict(orientation='horizontal', aspect=25,pad=0.15)
sckw = dict(s=110, c='darkviolet',edgecolor='w',linewidth=0.9, marker='*', zorder=10,label='Mooring')

xl = (132,268)
fg = plt.figure(figsize=(10, 4.5), dpi=300, tight_layout=True)
gs = fg.add_gridspec(1,2)

ax = plt.subplot(gs[0])
plt.contourf(lon1, dep1, maj1,np.arange(0, 0.21, 0.02),extend='max')
plt.colorbar(**cbdt)
plt.axvline(142, linestyle='--', color='w')
plt.plot(tray['x'].T, tray['z'].T, 'r--')
ax.invert_yaxis()
plt.xlim(xl)
plt.title(r'(a) Explained Variance of $\sf{U}$ (%)', fontdict=btxdt)
plt.gca().set_ylabel('Depth (m)',fontsize=16)


ax = plt.subplot(gs[1])
plt.pcolormesh(lon1, dep1, mph1, cmap='Paired',
               vmin=1, vmax=13, shading='nearest')
label = ['Feb', 'May', 'Aug', 'Nov']
cb = plt.colorbar(**cbdt)
cb.set_ticks(np.arange(2.5, 12, 3))
cb.set_ticklabels(label)
plt.axvline(142, linestyle='--', color='w')
plt.plot(tray['x'].T, tray['z'].T, 'r--')
ax.invert_yaxis()
plt.xlim(xl)
plt.title('(b) Phase of $\sf{U}$ (Month)', fontdict=btxdt)
plt.gca().yaxis.set_ticks_position('right')
plt.gca().yaxis.set_label_position('right')
plt.gca().set_ylabel('Depth (m)',fontsize=16)
#%%
plt.savefig('年调和分析.png', bbox_inches='tight')

