#
# LLM disclosure: Generated boilerplate/scaffolding file that does the following I asked:
# 1. Do a sweep over checkpoints x temperature x top_p
#
# I myself picked the range of temps and top_p
#

import csv
import os
from types import SimpleNamespace

from sonnet_generation import generate_submission_sonnets
from evaluation import test_sonnet

CHECKPOINTS = [
  '5-0.001-sonnet.pt',
  '5-0.003-sonnet.pt',
  '5-0.0001-sonnet.pt',
  '5-0.0003-sonnet.pt',
  '5-1e-05-sonnet.pt',
  '5-3e-05-sonnet.pt',
  '5-3e-06-sonnet.pt',
]


GOLD_PATH = 'data/TRUE_sonnets_held_out_dev.txt'
RESULTS_CSV = 'results_sonnet/sonnet_sweep.csv'
PRED_DIR = 'predictions/not_seeded'
FIELDS = ['checkpoint', 'temperature', 'top_p', 'chrf']
TEMPERATURES = []
TOP_PS = []

# Default
TEMPERATURES.extend([1.2])
TOP_PS.extend([0.9])

# Range
#TEMPERATURES.extend([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 2.0])
#TOP_PS.extend([0.85, 0.9, 0.95, 1.0])

# Extreme temp
#TEMPERATURES.extend([0.1, 1.0, 2.0, 10])
#TOP_PS.extend([0.9])

# Extreme top_p
#TEMPERATURES.extend([1.0])
#TOP_PS.extend([0.01, 0.1, 0.5])

TEMPERATURES = list(dict.fromkeys(TEMPERATURES))
TOP_PS = list(dict.fromkeys(TOP_PS))


def key(checkpoint, temperature, top_p):
  return (checkpoint, f'{temperature:g}', f'{top_p:g}')


def done_keys():
  if not os.path.exists(RESULTS_CSV):
    return set()
  with open(RESULTS_CSV) as f:
    return {(r['checkpoint'], r['temperature'], r['top_p']) for r in csv.DictReader(f)}


def append_row(checkpoint, temperature, top_p, chrf):
  os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
  new = not os.path.exists(RESULTS_CSV)
  with open(RESULTS_CSV, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
      w.writeheader()
    k = key(checkpoint, temperature, top_p)
    w.writerow(dict(checkpoint=k[0], temperature=k[1], top_p=k[2], chrf=f'{chrf:.4f}'))


if __name__ == '__main__':
  done = done_keys()
  for checkpoint in CHECKPOINTS:
    model_tag = os.path.splitext(os.path.basename(checkpoint))[0]
    for top_p in TOP_PS:
      for temperature in TEMPERATURES:
        if key(checkpoint, temperature, top_p) in done:
          print(f'skip {model_tag} temperature={temperature} top_p={top_p}')
          continue
        print(f'Evaluating {model_tag} temperature={temperature} top_p={top_p}')
        sonnet_out = f'{PRED_DIR}/{model_tag}_temp{temperature:g}_topp{top_p:g}.txt'
        args = SimpleNamespace(
          filepath=checkpoint,
          use_gpu=True,
          held_out_sonnet_path='data/sonnets_held_out_dev.txt',
          sonnet_out=sonnet_out,
          temperature=temperature,
          top_p=top_p,
        )
        generate_submission_sonnets(args)
        chrf_score = test_sonnet(test_path=sonnet_out, gold_path=GOLD_PATH)
        append_row(checkpoint, temperature, top_p, chrf_score)
        print(f'CHRF={chrf_score:.3f} | {model_tag} temperature={temperature} top_p={top_p}')