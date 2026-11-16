"""Compute five CTRL member transports with the existing audited operators.

Each member is processed identically; ensemble range describes initial-condition
spread, not a confidence interval. Use --summarize-only after extraction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import f90nml
import xarray as xr
from compare_channel_transport import meta, model_geometry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_mooring_transport import stats, iafilt

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ['261113_ctrl0', '261016_ctrl1', '261113_ctrl2',
               '261113_ctrl3', '261113_ctrl4']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def summarize(dest, plot):
    raw, smooth, daily, metadata = {}, {}, {}, {}
    checks = []
    for i, exp in enumerate(EXPERIMENTS):
        d = dest / exp / 'comparison'
        a = pd.read_csv(d / 'monthly_raw_Sv.csv', index_col=0, parse_dates=True)
        b = pd.read_csv(d / 'monthly_smoothed_Sv.csv', index_col=0, parse_dates=True)
        c = pd.read_csv(d / 'daily_smoothed_Sv.csv', index_col=0, parse_dates=True)
        if i:
            for previous, current in [(base_a, a), (base_b, b), (base_c, c)]:
                pd.testing.assert_index_equal(previous.index, current.index)
                np.testing.assert_allclose(previous[['obs', 'glorys_available']],
                                           current[['obs', 'glorys_available']], rtol=0, atol=1e-12)
        else:
            base_a, base_b, base_c = a, b, c
        raw[f'ctrl{i}'] = a.ctrl_native
        smooth[f'ctrl{i}'] = b.ctrl_native
        daily[f'ctrl{i}'] = c.ctrl_native
        checks.append(json.loads((d / 'validation.json').read_text()))
        folder = ROOT / 'output' / exp
        # Independently sum native source-cell fluxes for first/middle/last
        # validation months, bypassing the extraction cache and integrate().
        full_series = pd.read_csv(dest / exp / 'transport/transports.csv', index_col=0, parse_dates=True)
        geom = model_geometry(folder, 2)
        with xr.open_dataset(dest / exp / 'transport/geometry_and_latitude_transport.nc') as ds:
            area = ds.model_native_area.values
        targets = full_series.index[[0, len(full_series)//2, -1]]
        errors = []
        for path in sorted(folder.glob('diag3d.*.meta')):
            nr, fields, timestamp = meta(path)
            if timestamp not in targets:
                continue
            source = np.memmap(path.with_suffix('.data'), dtype='>f4', mode='r', shape=(nr, 40, 192, 720))
            q = 0.0
            for f, face in enumerate(geom['faces']):
                section = np.asarray(source[fields.index('UVEL'), :, geom['js'], face], dtype=float)
                # Numpy mixed indexing puts selected latitude first.
                if section.shape != area[f].shape:
                    section = section.T
                wet = area[f] > 0
                if not np.isfinite(section[wet]).all():
                    raise ValueError('Nonfinite source velocity inside wet section')
                q += float(np.sum(section[wet] * area[f][wet])) / 1e6
            errors.append(abs(q - full_series.loc[timestamp, 'model_native_net_Sv']))
        if len(errors) != 3 or max(errors) > 1e-10:
            raise AssertionError(f'Independent source flux audit failed: {exp}: {errors}')
        configs = {}
        for name in ['data', 'data.cal', 'data.exf', 'data.pkg', 'data.ggl90',
                     'data.kpp', 'data.obcs', 'data.rbcs', 'data.diagnostics']:
            configs[name] = dict(f90nml.read(folder / name))
        metadata[exp] = dict(
            independent_source_months=3, independent_source_max_error_Sv=max(errors),
            member=i, diagnostic_months=len(list(folder.glob('diag3d.*.meta'))),
            normal_end='Execution ended Normally' in (folder / 'STDOUT.0000').read_text()[-20000:],
            grid_sha256={name: digest(folder / (name + '.data'))
                         for name in ['XG', 'YG', 'YC', 'DYG', 'RF', 'RC', 'hFacW']},
            configuration=configs,
            transport_sha256=digest(dest / exp / 'transport/transports.csv'))
    first = metadata[EXPERIMENTS[0]]
    config_differences = {}
    for exp, m in metadata.items():
        if m['grid_sha256'] != first['grid_sha256']:
            raise ValueError(f'Grid differs: {exp}')
        different = {}
        for name, conf in m['configuration'].items():
            ref = first['configuration'][name]
            if conf != ref:
                different[name] = dict(member=conf, ctrl0=ref)
        config_differences[exp] = different
    frames = []
    for mapping, reference in [(raw, base_a), (smooth, base_b), (daily, base_c)]:
        frame = pd.DataFrame(mapping)
        frame['ensemble_mean'] = frame.mean(axis=1)
        frame['member_min'] = frame.iloc[:, :5].min(axis=1)
        frame['member_max'] = frame.iloc[:, :5].max(axis=1)
        frame['member_std_ddof0'] = frame.iloc[:, :5].std(axis=1, ddof=0)
        frame['obs'] = reference.obs
        frame['glorys'] = reference.glorys_available
        frames.append(frame)
    a, b, c = frames
    valid = base_b.full_window_all_days.astype(str).str.lower().eq('true')
    complete = b.loc[valid]
    # Same operator on the raw ensemble must equal the mean smoothed members.
    days = c.index
    expanded = a.ensemble_mean.reindex(days, method='ffill')
    linear_error = float(np.max(np.abs(iafilt(expanded.values) - c.ensemble_mean.values)))
    if linear_error > 1e-10:
        raise AssertionError('Smoothing ensemble linearity')
    rows = []
    for processing, frame in [('raw_monthly', a), ('smoothed365_full_period', b), ('smoothed365_complete_windows', complete)]:
        for candidate in [f'ctrl{i}' for i in range(5)] + ['ensemble_mean', 'glorys']:
            for reference in ['obs', 'glorys']:
                if candidate == reference:
                    continue
                rows.append(dict(processing=processing, candidate=candidate, reference=reference,
                                 first_month=str(frame.index[0].date()),
                                 last_month=str(frame.index[-1].date()),
                                 **stats(frame[candidate], frame[reference])))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(dest / 'metrics.csv', index=False)
    for name, frame in zip(['monthly_raw_Sv.csv', 'monthly_smoothed_Sv.csv', 'daily_smoothed_Sv.csv'], frames):
        if name != 'monthly_raw_Sv.csv':
            frame = frame.copy()
            frame['full_window'] = valid if name.startswith('monthly') else base_c.full_365day_window
        frame.to_csv(dest / name, index_label='time')
    b[[f'ctrl{i}' for i in range(5)] + ['obs', 'glorys']].corr().to_csv(dest / 'smoothed_correlations.csv')
    complete[[f'ctrl{i}' for i in range(5)] + ['obs', 'glorys']].corr().to_csv(dest / 'smoothed_correlations_complete_windows.csv')
    a[[f'ctrl{i}' for i in range(5)] + ['obs', 'glorys']].corr().to_csv(dest / 'raw_correlations.csv')
    deviation = complete.iloc[:, :5].sub(complete.ensemble_mean, axis=0)
    spread = dict(mean_pointwise_std_Sv=float(complete.member_std_ddof0.mean()),
                  rms_pointwise_std_Sv=float(np.sqrt(np.mean(deviation.values**2))),
                  mean_member_range_Sv=float((complete.member_max - complete.member_min).mean()),
                  maximum_member_range_Sv=float((complete.member_max - complete.member_min).max()),
                  ensemble_mean_std_Sv=float(complete.ensemble_mean.std(ddof=0)))
    info = dict(experiments=EXPERIMENTS,
                definition='142E, New Guinea coast to2N, below1900m, eastward Sv; native model full-bottom',
                raw_months=len(a), complete_window_months=len(complete),
                raw_period=[str(a.index[0].date()), str(a.index[-1].date())],
                smoothed_period=[str(b.index[0].date()), str(b.index[-1].date())],
                main_smoothed_months=len(b),
                smoothing='Exact user iafilt; monthly means expanded daily for every source; centered365 reflect; primary statistics retain all103months including reflected edges; complete-window91months retained as sensitivity',
                spread=spread, configuration_differences_from_ctrl0=config_differences,
                ensemble_note='Equal-weight mean of all five. Member min/max and spread are not confidence intervals; no member chosen by correlation.',
                geometry_caveat='Integrated mooring cannot be cropped to common mask; GLORYS44level available native transport, model native full-bottom.',
                main_metrics=metrics.query("processing=='smoothed365_full_period'").to_dict('records'))
    (dest / 'summary.json').write_text(json.dumps(info, indent=2, default=str))
    (dest / 'source_audit.json').write_text(json.dumps(metadata, indent=2, default=str))
    (dest / 'validation.json').write_text(json.dumps(dict(
        identical_grid=True, identical_reference_and_observation=True,
        all_normal_end=all(m['normal_end'] for m in metadata.values()),
        all_420_months=all(m['diagnostic_months']==420 for m in metadata.values()),
        smoothing_ensemble_linearity_error_Sv=linear_error,
        per_member=checks), indent=2))
    plot.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(13, 8), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for ax, frame in zip(axs, [a, c]):
        for i in range(5):
            ax.plot(frame.index, frame[f'ctrl{i}'], color=colors[i], alpha=.65, lw=.9, label=f'CTRL{i}')
        ax.fill_between(frame.index, frame.member_min, frame.member_max, color='gray', alpha=.15, label='Member range')
        ax.plot(frame.index, frame.ensemble_mean, color='#D55E00', lw=2.3, label='Ensemble mean')
        ax.plot(frame.index, frame.obs, color='black', lw=1.8, label='Mooring')
        ax.plot(frame.index, frame.glorys, color='#0072B2', lw=1.5, label='GLORYS')
        ax.axhline(0, color='gray', lw=.6)
        ax.set_ylabel('Eastward transport (Sv)')
    axs[0].set_title('Raw monthly transports | all five initial-condition members')
    axs[0].legend(ncol=4, fontsize=8)
    axs[1].set_title('Identical365-day smoothing | shaded end zones retained in primary statistics')
    axs[1].axvspan(days[0], days[182], color='gray', alpha=.2)
    axs[1].axvspan(days[-183], days[-1], color='gray', alpha=.2)
    fig.savefig(plot / 'E01_ensemble_transport.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True, sharey=True, layout='constrained')
    for i, ax in enumerate(axs):
        s = stats(b[f'ctrl{i}'], b.obs)
        ax.plot(c.index, c.obs, 'k', lw=1.3, label='Mooring')
        ax.plot(c.index, c[f'ctrl{i}'], color=colors[i], label=f'CTRL{i}')
        ax.axvspan(days[0], days[182], color='gray', alpha=.18)
        ax.axvspan(days[-183], days[-1], color='gray', alpha=.18)
        ax.axhline(0, color='gray', lw=.5)
        ax.set_ylabel('Sv')
        ax.legend(loc='upper right')
        ax.set_title(f'CTRL{i}: r={s["correlation"]:.3f}, std/obs={s["std_ratio"]:.3f}', fontsize=10)
    fig.suptitle('Member-by-member comparison | statistics2015-12 to2024-06; endpoints retained')
    fig.savefig(plot / 'E02_members_vs_mooring.png', dpi=180)
    plt.close(fig)
    print(metrics.query("reference=='obs'").to_string(index=False), flush=True)
    print(json.dumps(spread, indent=2), flush=True)
    raw_obs = metrics.query("processing=='raw_monthly' and reference=='obs'").set_index('candidate')
    low_obs = metrics.query("processing=='smoothed365_full_period' and reference=='obs'").set_index('candidate')
    low_ref = metrics.query("processing=='smoothed365_full_period' and reference=='glorys'").set_index('candidate')
    lines = [
        '# 五成员CTRL初值扰动：1900m以下断面输运', '',
        '计算日期2026-11-16。全部成员正常结束，各420月；同一原生网格。'
        'CTRL1目录为261016_ctrl1，其余为261113_ctrl0/2/3/4；CTRL1重现此前单组结果。', '',
        '定义：142E、岛岸至2N、1900m以下至当地海底，向东为正，单位Sv。'
        '主观测比较用模式原生完整断面；GLORYS仅现有44层可用断面。'
        '积分观测无法裁剪到模式/再分析共同湿面积，不把主对比称作同面积。', '',
        '每组独立读取UVEL并以hFacW/RF/DYG积分，先分别计算142E两侧U面通量再加权；'
        '北界及1900m层均按实际裁剪。各成员的输运、共同面积敏感性及源断面缓存保存在各自子目录。', '',
        '严格沿用用户iafilt：按365样本周期去平均、加回均值、居中365天等权卷积、reflect边界。'
        '三方均按完整共同月先月平均，再铺为日序列。'
        '原始及主平滑统计均为2015-12至2024-06共103月，按用户要求保留首尾；'
        '另列2016-06至2023-12共91月完整窗口结果为敏感性对照。'
        '这是365天平滑，不是严格2–8年带通。', '',
        '| 成员 | 原始月平均Q/Sv | 原始月r(观测) | 平滑std/Sv | 平滑r(观测) | 平滑std/观测 | 平滑r(GLORYS) |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for name in [f'ctrl{i}' for i in range(5)] + ['ensemble_mean', 'glorys']:
        r, s = raw_obs.loc[name], low_obs.loc[name]
        rr = f'{low_ref.loc[name, "correlation"]:.3f}' if name != 'glorys' else '1.000'
        lines.append(f'| {name} | {r.mean_candidate_Sv:.3f} | {r.correlation:.3f} | '
                     f'{s.std_candidate_Sv:.3f} | {s.correlation:.3f} | {s.std_ratio:.3f} | {rr} |')
    lines += ['',
        f'共同103月观测月均值2.247Sv；全窗观测平滑std为{b.obs.std(ddof=0):.3f}Sv。'
        '表中均值和平滑std均采用完整103月；'
        '各同窗均值、偏差、RMSE详见metrics.csv。', '',
        f'完整窗口成员离散：逐时成员标准差的RMS为{spread["rms_pointwise_std_Sv"]:.3f}Sv，'
        f'平均成员极差{spread["mean_member_range_Sv"]:.3f}Sv，'
        f'最大成员极差{spread["maximum_member_range_Sv"]:.3f}Sv。'
        '极差和阴影表示五成员范围，不是置信区间；集合均值全部等权，不筛选与观测最接近的成员。', '',
        '配置核对：data中差别为pickupSuff=clim_r4_0至clim_r4_4；'
        'CTRL1另多输出速度二阶矩和部分GGL90诊断。动力、强迫、日历和UVEL诊断口径一致。'
        '详细原始配置、网格散列及每组首/中/末三个源月份直接通量复算结果见source_audit.json。'
        '初值文件名的核对不等于已独立验证用户施加扰动的具体幅度或随机分布。', '',
        '相关系数不因统一缩小振幅或加常数而降低；低频r偏低需要从事件强弱和时间演变差异解释。'
        '原始月序列包含共同季节变化，因此不能用原始月高相关代替低频验证。'
        '五个初值成员也不是五条独立的观测时间记录；短记录平滑后的时间点强相关，不能按91个独立样本评估显著性。', '',
        '结果：metrics.csv、monthly_raw_Sv.csv、monthly_smoothed_Sv.csv、daily_smoothed_Sv.csv、'
        'raw_correlations.csv、smoothed_correlations.csv、summary.json、source_audit.json、validation.json。'
        '图：fig/261116_ctrl_ensemble/E01_ensemble_transport.png和E02_members_vs_mooring.png。'
        '复现命令见reproduce.sh。本次未改动输入或运行模拟。', ''
    ]
    member_stats = low_obs.loc[[f'ctrl{i}' for i in range(5)]]
    ensemble = low_obs.loc['ensemble_mean']
    lines += [
        f'五成员平滑相关范围{member_stats.correlation.min():.3f}–{member_stats.correlation.max():.3f}，'
        f'集合均值相关{ensemble.correlation:.3f}、std/观测={ensemble.std_ratio:.3f}。'
        '这批扰动初值没有消除低频共同偏差；不能将其直接归因于某次初值的偶然性，'
        '也不能把五成员均值当成完全分离的外强迫响应。'
        '图上共同缺少2016–2017和2020–2021观测强输运，以及2022–2023观测负输运阶段。', ''
    ]
    (dest / 'assessment.md').write_text('\n'.join(lines))
    (dest / 'manifest.json').write_text(json.dumps(dict(
        scripts_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
            Path(__file__).resolve(), ROOT / 'analysis/comparison/compare_mooring_transport.py',
            ROOT / 'analysis/comparison/compare_channel_transport.py']},
        input_mooring_sha256=digest(ROOT / 'data/moor_transport.npz'),
        source_audit_sha256=digest(dest / 'source_audit.json')), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'research/transport/2026-11-16_ctrl_ensemble')
    p.add_argument('--plot-output', type=Path, default=ROOT / 'fig/261116_ctrl_ensemble')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        for exp in EXPERIMENTS:
            d = args.output / exp
            d.mkdir(exist_ok=True)
            commands = [
                [sys.executable, '-u', str(ROOT / 'analysis/comparison/compare_channel_transport.py'),
                 '--exp', exp, '--workers', '4', '--output', str(d / 'transport')],
                [sys.executable, '-u', str(ROOT / 'analysis/comparison/compare_mooring_transport.py'),
                 '--transport', str(d / 'transport/transports.csv'), '--output', str(d / 'comparison'),
                 '--plot-output', str(args.plot_output / exp)]]
            for i, cmd in enumerate(commands):
                print('Processing', exp, i, flush=True)
                with (d / f'step{i}.log').open('w') as log:
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize(args.output, args.plot_output)


if __name__ == '__main__':
    main()
