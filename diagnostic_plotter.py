#!/usr/bin/env python3
"""Collision diagnostics: fixed z projection by default; full-vector validity checks retained.
Dependencies: numpy, pandas, matplotlib (only for plotting).
Example: python diagnostic_plotter.py --input momentum_history.csv --start-time 1.6
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

AXES = 'xyz'

def vector(df, prefix):
    names = [prefix + '_' + a for a in AXES]
    if all(c in df for c in names):
        return df[names].to_numpy(dtype=float)
    return np.full((len(df), 3), np.nan)

def lengths(v):
    return np.linalg.norm(v, axis=1)

def stats(v, mask):
    a = np.asarray(v)[mask & np.isfinite(v)]
    return ('unavailable' if not len(a) else
            'max={:.9e}, RMS={:.9e}, n={}'.format(np.max(np.abs(a)), np.sqrt(np.mean(a*a)),len(a)))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='momentum_history.csv')
    ap.add_argument('--output-dir', default='momentum_diagnostics_z')
    ap.add_argument('--start-time', type=float, default=1.6,
                    help='Post-inlet anchor time (default 1.6); choose a time before collision')
    ap.add_argument('--end-time', type=float, default=None)
    ap.add_argument('--force-tolerance', type=float, default=1e-10)
    ap.add_argument('--balance-tolerance', type=float, default=1e-8,
                    help='Absolute vector rate-error tolerance')
    ap.add_argument('--impulse-tolerance', type=float, default=1e-10)
    ap.add_argument('--direction', nargs=3, type=float, default=[0., 0., 1.],
                    help='Fixed projection direction; default 0 0 1, never inferred from forcing')
    ap.add_argument('--external-tolerance', type=float, default=1e-12)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    for c in ['time', 'P_total_x', 'P_total_y', 'P_total_z']:
        if c not in df:
            raise ValueError('Missing required column: ' + c)
    t = df.time.to_numpy(float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Need >=2 finite strictly increasing times; duplicate rows must be resolved explicitly')
    dt = np.r_[np.nan, np.diff(t)]
    p = vector(df, 'P_total')
    delta = np.vstack([np.full(3, np.nan), np.diff(p, axis=0)])
    j = vector(df, 'J_expected')
    raw = delta - j
    rate = raw / dt[:, None]
    f = vector(df, 'F_expected')
    expected_rate = j / dt[:, None]
    measured_rate = delta / dt[:, None]
    coupling = vector(df, 'F_dpd_to_dem') + vector(df, 'F_dem_to_dpd')
    cn = lengths(coupling)
    if args.direction is None:
        finite = np.isfinite(f).all(axis=1)
        direction = np.mean(f[finite], axis=0) if finite.any() else np.array([1.,0.,0.])
        if np.linalg.norm(direction) == 0:
            direction = np.array([1.,0.,0.])
    else:
        direction = np.array(args.direction)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('Direction must be a finite nonzero vector')
    direction = direction / np.linalg.norm(direction)
    start = t[0] if args.start_time is None else args.start_time
    end = t[-1] if args.end_time is None else args.end_time
    anchors = np.flatnonzero((t >= start - 1e-12) & (t <= end + 1e-12))
    if len(anchors) < 2:
        raise ValueError('Selected window must contain at least two output times')
    anchor = anchors[0]
    window = (np.arange(len(t)) > anchor) & (t <= end + 1e-12)
    flags = ['balance_valid','every_step_sampled','population_unchanged','force_budget_complete']
    strict = window & np.isfinite(raw).all(axis=1)
    for flag in flags:
        strict &= df[flag].eq(1).to_numpy() if flag in df else False
    exploratory = window & np.isfinite(raw).all(axis=1)
    for flag in ['population_unchanged','force_budget_complete']:
        exploratory &= df[flag].eq(1).to_numpy() if flag in df else False
    report = ['DEM-DPD momentum diagnostics', '='*32,
              'Rows: {}; window anchor: {:.16g}; end: {:.16g}'.format(len(df), t[anchor], end),
              'Fixed projection direction: ' + str(direction.tolist()),
              'Strict valid intervals: {}'.format(int(strict.sum())),
              'Exploratory intervals (sampling flag NOT overridden): {}'.format(int(exploratory.sum()))]
    for flag in flags:
        report.append('{}: {}'.format(flag, df[flag].value_counts(dropna=False).to_dict() if flag in df else 'MISSING'))
    if 'step' in df and df.step.nunique() == 1:
        report.append('WARNING: step is constant. A consecutive-STEP test cannot establish per-step sampling.')
    report += ['Strict impulse norm: ' + stats(lengths(raw), strict),
               'Strict rate-error norm: ' + stats(lengths(rate), strict),
               'Exploratory impulse norm: ' + stats(lengths(raw), exploratory),
               'Exploratory rate-error norm: ' + stats(lengths(rate), exploratory),
               'Coupling norm (selected window): ' + stats(cn, window)]
    if strict.any():
        passed = (lengths(raw)[strict].max() <= args.impulse_tolerance and
                  lengths(rate)[strict].max() <= args.balance_tolerance)
        report.append('GLOBAL STATUS: ' + ('PASS' if passed else 'CHECK') + ' on strict valid intervals only')
    else:
        report.append('GLOBAL STATUS: UNAVAILABLE (no strict valid intervals); exploratory numbers are not certification.')
    if exploratory.any():
        duration = dt[exploratory].sum()
        total_dp = delta[exploratory].sum(axis=0)
        total_j = j[exploratory].sum(axis=0)
        report += ['Exploratory duration: {:.16e}'.format(duration),
                   'Exploratory measured projected mean rate: {:.16e}'.format(total_dp @ direction / duration),
                   'Exploratory expected projected mean rate: {:.16e}'.format(total_j @ direction / duration),
                   'Exploratory cumulative projected mismatch: {:.16e}'.format((total_dp-total_j) @ direction),
                   'Exploratory cumulative vector mismatch: {:.16e}'.format(np.linalg.norm(total_dp-total_j))]
    if 'population_unchanged' in df:
        report.append('Population/mass-change interval END times: ' + str(t[df.population_unchanged.eq(0)].tolist()))
    selected_rows = (np.arange(len(t)) >= anchor) & (t <= end + 1e-12)
    ef = lengths(f[selected_rows])
    recorded_unforced = len(ef) > 0 and np.isfinite(ef).all() and np.max(ef) <= args.external_tolerance
    report.append('Recorded expected force zero throughout selected window: ' + str(recorded_unforced))
    if not recorded_unforced:
        report.append('WARNING: not a verified unforced collision window. Plotting cannot disable simulation forces.')
    if len(ef) and np.isfinite(ef).any():
        report.append('Maximum recorded expected force norm: {:.9e}'.format(np.nanmax(ef)))
    for prefix in ('P_dem', 'P_dpd', 'P_total'):
        vv = vector(df,prefix)
        change = vv[anchors[-1]]-vv[anchor]
        report.append(prefix + ' final-minus-anchor [x,y,z]: ' + str(change.tolist()))
    reference = np.full_like(p, np.nan)
    reference[anchor] = p[anchor]
    for i in range(anchor+1, len(df)):
        if t[i] > end + 1e-12:
            break
        if not exploratory[i] or not np.isfinite(reference[i-1]).all():
            continue
        reference[i] = reference[i-1] + j[i]
    audit = df.copy()
    audit['selected_interval'] = window.astype(int)
    audit['strict_valid_recomputed'] = strict.astype(int)
    audit['exploratory_interval'] = exploratory.astype(int)
    audit['interval_start_time'] = t - dt
    audit['recomputed_impulse_error_norm'] = lengths(raw)
    audit['recomputed_rate_error_norm'] = lengths(rate)
    audit['strict_impulse_error_norm'] = np.where(strict, lengths(raw), np.nan)
    audit['delta_P_dem_projected'] = np.r_[np.nan, np.diff(vector(df, 'P_dem') @ direction)]
    audit['delta_P_dpd_projected'] = np.r_[np.nan, np.diff(vector(df, 'P_dpd') @ direction)]
    audit['delta_P_total_projected'] = delta @ direction
    for i,a in enumerate(AXES):
        audit['recomputed_impulse_error_'+a] = raw[:,i]
        audit['reference_momentum_'+a] = reference[:,i]
    audit.to_csv(out / 'interval_audit.csv', index=False)
    text = '\n'.join(report) + '\n'
    (out / 'momentum_diagnostic_summary.txt').write_text(text, encoding='utf-8')
    print(text)
    if args.no_plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(name, ylabel, title):
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / name, dpi=200)
        plt.close()
    plt.figure(figsize=(10,6))
    for prefix,label in [('P_dem','DEM'),('P_dpd','DPD'),('P_total','Total')]:
        plt.plot(t, vector(df,prefix) @ direction, label=label)
    plt.plot(t, reference @ direction, '--', label='Integrated external impulse reference')
    if recorded_unforced:
        plt.plot(t[selected_rows], np.full(selected_rows.sum(),p[anchor] @ direction), ':', label='Constant post-inlet total momentum')
    save('01_total_momentum_vs_time.png','Momentum along fixed direction (default z)','Collision momentum and post-inlet reference')
    plt.figure(figsize=(10,6))
    for prefix,label in [('F_dpd_to_dem','DPD to DEM'),('F_dem_to_dpd','DEM to DPD')]:
        plt.plot(t, vector(df,prefix) @ direction, label=label)
    plt.plot(t, coupling @ direction, 'k-',label='Coupling sum')
    save('02_coupling_force_balance.png','Projected force','Recorded coupling action-reaction')
    plt.figure(figsize=(10,6))
    positive = np.isfinite(cn) & (cn > 0)
    plt.semilogy(t, np.where(positive,cn,np.nan), 'o-',label='Nonzero coupling residuals')
    plt.axhline(args.force_tolerance,color='red',ls='--',label='Tolerance')
    plt.xlim(t[0],t[-1])
    plt.text(.02,.03,'Exact-zero samples: {}'.format(int(np.sum(cn==0))),transform=plt.gca().transAxes)
    save('03_coupling_residual.png','Force residual norm','Coupling residual (zeros omitted, not floored)')
    plt.figure(figsize=(10,6))
    plt.plot(t, measured_rate @ direction,label='Interval momentum rate')
    plt.plot(t, expected_rate @ direction,'--',label='Recorded impulse / interval duration')
    plt.plot(t, f @ direction,':',label='Endpoint expected force (not interval average)')
    save('04_global_momentum_balance.png','Projected force / rate','Interval-average momentum budget; inspect validity flags')
    plt.figure(figsize=(10,6))
    plt.plot(t,np.where(exploratory,lengths(rate),np.nan),'o--',label='Exploratory rate error')
    plt.plot(t,np.where(strict,lengths(rate),np.nan),'o',label='Strict valid rate error')
    plt.axhline(args.balance_tolerance,color='red',ls='--',label='Rate tolerance')
    save('05_global_residual.png','Vector rate-error norm','Global error: strict vs unverified sampling')
    plt.figure(figsize=(10,6))
    for prefix,label in [('P_dem','DEM change'),('P_dpd','DPD change'),('P_total','Total change')]:
        vv = vector(df,prefix) @ direction
        plt.plot(t[selected_rows], (vv-vv[anchor])[selected_rows], label=label)
    plt.plot(t[selected_rows], ((reference-p[anchor]) @ direction)[selected_rows], '--', label='Expected total change from external impulse')
    save('07_momentum_changes.png','Momentum change from anchor (default z)','Equal-and-opposite transfer test')
    plt.figure(figsize=(10,6))
    for i,a in enumerate(AXES):
        drift = p[:,i]-p[anchor,i]
        plt.plot(t[selected_rows],drift[selected_rows],label='Total momentum change '+a)
    save('08_total_momentum_components.png','Momentum change from anchor','All components: constant if no external impulse')
    plt.figure(figsize=(10,6))
    for prefix,label in [('P_dem','DEM interval change'),('P_dpd','DPD interval change')]:
        change=np.r_[np.nan,np.diff(vector(df,prefix) @ direction)]
        plt.plot(t[window],change[window],'o-',label=label)
    plt.plot(t[window],(delta @ direction)[window],'k--',label='Sum of interval changes')
    plt.plot(t[window],(j @ direction)[window],':',label='Expected external impulse')
    save('09_interval_transfer.png','Interval momentum change (default z)','Particle-by-particle impulse transfer')
    plt.figure(figsize=(10,6))
    for flag in flags:
        if flag in df:
            plt.step(t,df[flag],where='post',label=flag)
    save('06_validity_flags.png','Flag (0/1)','Diagnostic completeness and validity')

if __name__ == '__main__':
    main()