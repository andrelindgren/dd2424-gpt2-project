#!/usr/bin/env python3

#
# LLM disclosure: Iteratively generated graphing and summerizing file that does the following I asked:
#
# 1. Read results/runs.csv and give a table of best results for each run
# 2. Show mean acc with standard deviation for the seeded runs
# 3. Show best config (mean acc) for the extended head runs
# 4. Create a graph that shows cumulative hyperparameter search time in chronological order
#     for each model config (head, full, top-2/4/8, head extended)
# 5. Add a bar that shows the noise from the dataset. (The 95% one config is definitely better then the other)
#
# I added the min_improvement() formula:
#   1.96 * math.sqrt(2 * acc * (1 - acc) / n)
# for the calculation of noise from the dataset

'''
Reads results/runs.csv and produces:
  results/anytime_sst.png   best dev acc so far vs cumulative search time, one line per arm
  a printed summary per arm, and the dev-set noise floor

  python analysis.py
'''
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS_CSV = 'results/runs.csv'
ARM_ORDER = ['head', 'head_ext', 'top2', 'top4', 'top8', 'full']
# stages that count as search effort for the anytime curve
SEARCH_STAGES = {'extract', 'lr', 'ladder', 'reg', 'budget'}
# arms whose runs are grouped into multi-seed configs; the search objective is the mean
MEAN_SEED_ARMS = {'head_ext'}

def load(dataset):
  with open(RUNS_CSV) as f:
    return [r for r in csv.DictReader(f) if r['dataset'] == dataset]


def config_id(run_name):
  '''Seeds of one config share a name up to the --seedN suffix.'''
  return run_name.split('--seed')[0]


def anytime(rows):
  '''
  Cumulative wall-clock vs running best dev accuracy, per arm, in logged order.
  '''
  curves = defaultdict(lambda: ([], []))
  clock = defaultdict(float)
  best = defaultdict(float)
  pending = {}      # arm -> (config_id, [accs])

  def flush(arm):
    if arm in pending and pending[arm][1]:
      accs = pending[arm][1]
      best[arm] = max(best[arm], sum(accs) / len(accs))
      curves[arm][0].append(clock[arm] / 60.0)
      curves[arm][1].append(best[arm])
    pending.pop(arm, None)

  for r in rows:
    if r['stage'] not in SEARCH_STAGES:
      continue
    arm = r['arm']
    clock[arm] += float(r['wall_seconds'])

    if arm in MEAN_SEED_ARMS:
      cid = config_id(r['run_name'])
      if arm in pending and pending[arm][0] != cid:
        flush(arm)
      pending.setdefault(arm, (cid, []))
      if r['best_dev_acc']:
        pending[arm][1].append(float(r['best_dev_acc']))
      continue

    if r['best_dev_acc']:                      # extraction rows have no accuracy
      best[arm] = max(best[arm], float(r['best_dev_acc']))
    if best[arm] > 0:
      curves[arm][0].append(clock[arm] / 60.0)
      curves[arm][1].append(best[arm])

  for arm in list(pending):
    flush(arm)
  return curves


def plot_anytime(curves, path, ruler=None):
  fig, ax = plt.subplots(figsize=(6.6, 4))
  for arm in ARM_ORDER:
    if arm not in curves:
      continue
    x, y = curves[arm]
    ax.step(x, y, where='post', marker='o', markersize=3, label=arm)
  ax.set_xlabel('cumulative search time (min)')
  ax.set_ylabel('best dev accuracy so far')
  ax.grid(alpha=0.3)

  if ruler:
    label, delta = ruler
    x0, x1 = ax.get_xlim()
    pad = 0.05 * (x1 - x0)
    ax.set_xlim(x0 - pad, x1)          # room to the left of the first point
    y0, y1 = ax.get_ylim()
    ax.errorbar(x0 - pad / 2, (y0 + y1) / 2, yerr=delta / 2,
                color='0.35', lw=1.4, capsize=4,
                label=f'dev-set noise ({delta:.3f})')

  ax.legend(loc='lower right', fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'wrote {path}')


def min_improvement(acc, n):
  '''
  Smallest dev-acc difference between two runs that is not just dev-set noise.

  Binomial standard error = sqrt(acc*(1-acc)/n)
  Multiplied by 2 for diff between 2 runs.
  1.96 = 95% confidence

  Assumes the errors are not correlated between models.
  But most probably there is a large correlation between which
  things the two models get right and wrong. So this is an upper bound.
  '''
  return 1.96 * math.sqrt(2 * acc * (1 - acc) / n)


def summarise(rows, dataset):
  print(f'\n=== {dataset} ===')
  by_arm = defaultdict(list)
  for r in rows:
    if r['best_dev_acc']:
      by_arm[r['arm']].append(r)

  print(f'{"arm":8} {"runs":>5} {"best":>7} {"lr":>9} {"steps":>8} {"s/run":>8} {"mem GB":>7}')
  for arm in ARM_ORDER:
    if arm not in by_arm:
      continue
    rs = by_arm[arm]
    b = max(rs, key=lambda r: float(r['best_dev_acc']))
    secs = sum(float(r['wall_seconds']) for r in rs) / len(rs)
    mem = max(int(float(r['peak_mem_bytes'] or 0)) for r in rs) / 1e9
    print(f'{arm:8} {len(rs):5d} {float(b["best_dev_acc"]):7.4f} {float(b["lr"]):9.1e} '
          f'{int(b["opt_steps"]):8d} {secs:8.1f} {mem:7.2f}')

  # matched-budget head search: rank configs by seed mean, not by best single run
  ext = [r for r in rows if r['arm'] in MEAN_SEED_ARMS and r['best_dev_acc']]
  if ext:
    by_cfg = defaultdict(list)
    for r in ext:
      by_cfg[config_id(r['run_name'])].append(r)
    means = {}
    for cid, rs in by_cfg.items():
      accs = [float(r['best_dev_acc']) for r in rs]
      means[cid] = (sum(accs) / len(accs), accs, rs[0])
    best_cid = max(means, key=lambda c: means[c][0])
    mean, accs, r0 = means[best_cid]
    spread = max(accs) - min(accs)
    single_max = max(float(r['best_dev_acc']) for r in ext)
    total = sum(float(r['wall_seconds']) for r in ext)
    print(f'\nmatched-budget head search: {len(by_cfg)} configs, {len(ext)} runs, {total / 60:.1f} min')
    print(f'  best config (seed mean): {mean:.4f} over {len(accs)} seeds, spread {spread:.4f}')
    print(f'  lr {float(r0["lr"]):.2e}, dropout {r0["dropout"]}, wd {r0["weight_decay"]}, '
          f'epochs {r0["epochs"]}, best_epoch {r0["best_epoch"]}')
    print()

  # seed spread
  print("Seed spread for best models")
  for arm in ARM_ORDER:
    seeds = [float(r['best_dev_acc']) for r in rows
             if r['arm'] == arm and r['stage'] == 'seed' and r['best_dev_acc']]
    if len(seeds) > 1:
      mean = sum(seeds) / len(seeds)
      std = math.sqrt(sum((s - mean) ** 2 for s in seeds) / (len(seeds) - 1))
      print(f'{arm}: {len(seeds)} seeds, dev acc {mean:.4f} +/- {std:.4f} (std)')

  # single threshold: differences smaller than this are not proven
  n_dev = dev_size(dataset)
  best = max(float(r['best_dev_acc']) for r in rows if r['best_dev_acc'])
  thr = min_improvement(best, n_dev)
  print(f'\ndev n={n_dev}: two runs differing by less than {thr:.3f} '
        f'can be considered equal')
  return thr


def dev_size(dataset):
  path = {'sst': 'data/ids-sst-dev.csv', 'cfimdb': 'data/ids-cfimdb-dev.csv'}[dataset]
  with open(path, encoding='utf-8') as f:
    return sum(1 for _ in csv.DictReader(f, delimiter='\t'))


if __name__ == '__main__':
  for dataset in ('sst', 'cfimdb'):
    rows = load(dataset)
    if not rows:
      continue
    thr = summarise(rows, dataset)
    if dataset == 'sst':
      plot_anytime(anytime(rows), 'results/anytime_sst.png',
                   ('single runs', thr) if thr else None)
