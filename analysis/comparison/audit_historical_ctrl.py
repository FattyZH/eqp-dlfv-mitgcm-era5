"""Read-only Git/config audit of the last commit preceding2026-06-03."""
from pathlib import Path
import subprocess,json,hashlib,difflib
import f90nml
ROOT=Path(__file__).resolve().parents[2]
REV='3ed466ddaea13783bd59a91fe4f01fec1e151e0c'
DEST=ROOT/'research/model_history/2026-11-16_june03_vs_ctrl'
CURRENT=ROOT/'output/261016_ctrl1'
def git(*args):
 return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 differences={};manifest={}
 for name in ['data','data.ggl90','data.pkg','data.cal','data.obcs','data.exf']:
  oldpath='config/data.exf.jra55_d' if name=='data.exf' else 'input/'+name
  before=git('show',REV+':'+oldpath);after=(CURRENT/name).read_text()
  for tag,value in [('june01',before),('current_ctrl1',after)]:
   p=DEST/tag/name;p.parent.mkdir(exist_ok=True);p.write_text(value)
   manifest[str(p.relative_to(DEST))]=hashlib.sha256(value.encode()).hexdigest()
  (DEST/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=REV[:7]+':'+oldpath,tofile='output/261016_ctrl1/'+name)))
  a=f90nml.reads(before);b=f90nml.reads(after);items=[]
  for group in sorted(set(a)|set(b)):
   for key in sorted(set(a.get(group,{}))|set(b.get(group,{}))):
    x=a.get(group,{}).get(key);y=b.get(group,{}).get(key)
    if x!=y:items.append(dict(group=group,parameter=key,june01_explicit=x,current_explicit=y))
  differences[name]=items
 for name in ['SIZE.h','CPP_OPTIONS.h','EXF_OPTIONS.h','OBCS_OPTIONS.h','packages.conf']:
  (DEST/(name+'.diff')).write_text(git('diff',REV,'HEAD','--','code/'+name))
 log=git('log','--all','--since=2026-05-25','--until=2026-06-12','--date=iso','--format=%H %ad %s')
 (DEST/'git_history.txt').write_text(log)
 (DEST/'historical_symlinks.txt').write_text('\n'.join(p+': '+git('show',REV+':'+p) for p in ['input/data.exf','input/exf']))
 result=dict(requested_date='2026-06-03',baseline_commit=REV,baseline_commit_date='2026-06-01',no_commit_on_requested_date=True,current_run='output/261016_ctrl1',explicit_namelist_differences=differences,snapshot_sha256=manifest,limits=['Git snapshot does not prove actual June03 run configuration or correlation','Historical forcing/initial/bathymetry/OBCS binaries are not recovered by Git','Historical omitted defaults are inferred from available MITgcm source, not historical executable','Source CPP diff does not establish flags in either compiled executable'])
 (DEST/'parameter_differences.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('Wrote',DEST)
if __name__=='__main__':main()
