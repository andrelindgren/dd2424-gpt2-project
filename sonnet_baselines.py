#
# LLM Disclosure: Generated functions in order to do my three proposed tests:
#
# 1. Prompt only: What if we dont output anything more then the seed prompt we got
# 2. Wrong sonnet: What if we supply another shakespare sonnet, what will our score be?
# 3. Prompt + wrong tail of sonnet: Same as 2 but with correct prompt at the start
#
# Then extended by LLM generated functions that investigates:
# 4. How the capped output of 128 characters (because of max_length=128) effect the benchmark.
# 5. How random characters would score.

import os
import random
import statistics

from datasets import SonnetsDataset
from evaluation import test_sonnet
from transformers import GPT2Tokenizer


GOLD = 'data/TRUE_sonnets_held_out_dev.txt'
PROMPTS = 'data/sonnets_held_out_dev.txt'
OUT_DIR = 'sonnet_baselines'
N_SHUFFLES = 20
PROMPT_LINES = 3


# --- helpers -----------------------------------------------------------------

def load(path):
  """[(id, text), ...] exactly as the evaluation pipeline sees it."""
  return [(x[0], x[1]) for x in SonnetsDataset(path)]


def write_submission(path, items):
  """Byte-identical layout to generate_submission_sonnets."""
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w+', encoding='utf-8') as f:
    f.write('--Generated Sonnets-- \n\n')
    for sonnet_id, text in items:
      f.write(f'\n{sonnet_id}\n')
      f.write(f'{text}\n\n')
  return path


def score(test_path, gold_path=GOLD):
  return test_sonnet(test_path=test_path, gold_path=gold_path)


def derangement(n, rng):
  """Permutation with no fixed point, so nothing is paired with itself."""
  if n < 2:
    raise ValueError('need at least 2 sonnets to derange')
  while True:
    perm = list(range(n))
    rng.shuffle(perm)
    if all(i != p for i, p in enumerate(perm)):
      return perm


def split_lines(text, k=PROMPT_LINES):
  lines = text.split('\n')
  return lines[:k], lines[k:]


# --- baselines ---------------------------------------------------------------


def prompt_only(gold):
  """Emit the 3 seed lines and stop. The free score for copying the prompt."""
  items = [(sid, '\n'.join(split_lines(t)[0])) for sid, t in gold]
  path = write_submission(f'{OUT_DIR}/prompt_only.txt', items)
  return score(path)


def shuffled_gold(gold, n=N_SHUFFLES):
  """Real Shakespeare against the wrong real Shakespeare."""
  scores = []
  for seed in range(n):
    perm = derangement(len(gold), random.Random(seed))
    items = [(gold[i][0], gold[p][1]) for i, p in enumerate(perm)]
    path = write_submission(f'{OUT_DIR}/shuffled_{seed}.txt', items)
    scores.append(score(path))
  return scores


def prompt_plus_wrong_tail(gold, n=N_SHUFFLES):
  """The realistic floor: correct 3 seed lines, then a different sonnet's tail."""
  scores = []
  for seed in range(n):
    perm = derangement(len(gold), random.Random(1000 + seed))
    items = []
    for i, p in enumerate(perm):
      head = split_lines(gold[i][1])[0]
      tail = split_lines(gold[p][1])[1]
      items.append((gold[i][0], '\n'.join(head + tail)))
    path = write_submission(f'{OUT_DIR}/prompt_wrong_tail_{seed}.txt', items)
    scores.append(score(path))
  return scores


# --- capped baselines --------------------------------------------------------
#
# All real generations used generate(max_length=128), which in this repo means
# 128 NEW tokens appended after the prompt. To compare floors against those
# generations fairly, each floor is re-emitted under the same budget.

MAX_NEW_TOKENS = 128


def gold_cut_at_generation_budget(gold, tokenizer, max_new_tokens=128, prompt_lines=PROMPT_LINES):
  """Real Shakespeare, tail truncated the way this specific generate() loop
  would cut it: prompt is untouched, only the continuation is capped at
  max_new_tokens tokens (this repo's max_length param == max_new_tokens,
  since the loop appends one token per iteration on top of the prompt)."""
  items = []
  for sid, text in gold:
    head_lines, tail_lines = split_lines(text, prompt_lines)
    head = '\n'.join(head_lines)
    tail = '\n'.join(tail_lines)
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)[:max_new_tokens]
    tail_cut = tokenizer.decode(tail_ids)
    items.append((sid, head + '\n' + tail_cut))
  path = write_submission(f'{OUT_DIR}/gold_cut_tail_{max_new_tokens}.txt', items)
  return score(path)


def cap_tail(tail, tokenizer, max_new_tokens=MAX_NEW_TOKENS):
  """Cut a continuation to the generation loop's new-token budget."""
  ids = tokenizer.encode(tail, add_special_tokens=False)[:max_new_tokens]
  return tokenizer.decode(ids)


def prompt_only_capped(gold, tokenizer, max_new_tokens=MAX_NEW_TOKENS):
  """Identical to prompt_only: 3 seed lines emit zero new tokens, so the cap
  cannot bind. Included only so every row in the table is generated the same way."""
  return prompt_only(gold)


def shuffled_gold_capped(gold, tokenizer, n=N_SHUFFLES, max_new_tokens=MAX_NEW_TOKENS):
  """Wrong real sonnet, truncated to the same total emitted length a real
  generation for that item would have: its own prompt's token count + budget."""
  scores = []
  for seed in range(n):
    perm = derangement(len(gold), random.Random(seed))
    items = []
    for i, p in enumerate(perm):
      own_head = '\n'.join(split_lines(gold[i][1])[0])
      n_head = len(tokenizer.encode(own_head, add_special_tokens=False))
      wrong = gold[p][1]
      ids = tokenizer.encode(wrong, add_special_tokens=False)[:n_head + max_new_tokens]
      items.append((gold[i][0], tokenizer.decode(ids)))
    path = write_submission(f'{OUT_DIR}/shuffled_capped_{seed}.txt', items)
    scores.append(score(path))
  return scores


def prompt_plus_wrong_tail_capped(gold, tokenizer, n=N_SHUFFLES, max_new_tokens=MAX_NEW_TOKENS):
  """The floor that matches your generations exactly in shape: correct 3 seed
  lines (free), then a different sonnet's tail capped at the new-token budget."""
  scores = []
  for seed in range(n):
    perm = derangement(len(gold), random.Random(1000 + seed))
    items = []
    for i, p in enumerate(perm):
      head = '\n'.join(split_lines(gold[i][1])[0])
      tail = '\n'.join(split_lines(gold[p][1])[1])
      items.append((gold[i][0], head + '\n' + cap_tail(tail, tokenizer, max_new_tokens)))
    path = write_submission(f'{OUT_DIR}/prompt_wrong_tail_capped_{seed}.txt', items)
    scores.append(score(path))
  return scores



# --- random-character floor --------------------------------------------------


RANDOM_ALPHABET = (
  'abcdefghijklmnopqrstuvwxyz'
  'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  '          '   
  '\n'
  ",.;:'!?-"
)


def gold_char_alphabet(gold, prompt_lines=PROMPT_LINES):
  """Alternative alphabet: every character of every gold tail, so sampling
  reproduces Shakespeare's unigram character frequencies and nothing else."""
  return ''.join('\n'.join(split_lines(t, prompt_lines)[1]) for _, t in gold)


def random_text(n_chars, alphabet, rng):
  return ''.join(rng.choice(alphabet) for _ in range(n_chars))


def random_chars_only(gold, n=N_SHUFFLES, alphabet=RANDOM_ALPHABET, prompt_lines=PROMPT_LINES):
  """Absolute floor: no prompt, no words, just noise at the same character
  length as the full gold sonnet."""
  scores = []
  for seed in range(n):
    rng = random.Random(2000 + seed)
    items = [(sid, random_text(len(text), alphabet, rng)) for sid, text in gold]
    path = write_submission(f'{OUT_DIR}/random_chars_{seed}.txt', items)
    scores.append(score(path))
  return scores


def prompt_plus_random_tail(gold, n=N_SHUFFLES, alphabet=RANDOM_ALPHABET, prompt_lines=PROMPT_LINES):
  """Correct 3 seed lines, then noise of the same character length as the real
  tail. Isolates what the prompt alone is worth versus what a tail must beat."""
  scores = []
  for seed in range(n):
    rng = random.Random(3000 + seed)
    items = []
    for sid, text in gold:
      head_lines, tail_lines = split_lines(text, prompt_lines)
      tail = '\n'.join(tail_lines)
      items.append((sid, '\n'.join(head_lines) + '\n' + random_text(len(tail), alphabet, rng)))
    path = write_submission(f'{OUT_DIR}/prompt_random_tail_{seed}.txt', items)
    scores.append(score(path))
  return scores


def prompt_plus_random_tail_capped(gold, tokenizer, n=N_SHUFFLES, alphabet=RANDOM_ALPHABET,
                                   max_new_tokens=MAX_NEW_TOKENS, prompt_lines=PROMPT_LINES):
  """Same, but the noise is generated token-by-token under the real budget:
  128 new tokens, then stop. This is what your generate() loop would actually
  have emitted if the model had sampled uniformly."""
  scores = []
  for seed in range(n):
    rng = random.Random(4000 + seed)
    items = []
    for sid, text in gold:
      head_lines, tail_lines = split_lines(text, prompt_lines)
      tail = '\n'.join(tail_lines)
      # overshoot in characters, then cut to the token budget
      noise = random_text(len(tail) * 4, alphabet, rng)
      ids = tokenizer.encode(noise, add_special_tokens=False, verbose=False)[:max_new_tokens]
      items.append((sid, '\n'.join(head_lines) + '\n' + tokenizer.decode(ids)))
    path = write_submission(f'{OUT_DIR}/prompt_random_tail_capped_{seed}.txt', items)
    scores.append(score(path))
  return scores


# --- main --------------------------------------------------------------------


def summarize(name, scores):
  mean = statistics.mean(scores)
  sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
  print(f'{name:<28} {mean:6.2f}  +/- {sd:4.2f}   '
        f'[{min(scores):.2f}, {max(scores):.2f}]  n={len(scores)}')
  return mean


if __name__ == '__main__':
  gold = load(GOLD)
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

  print('=== ceiling ===')
  print(f'{"identity (gold vs gold)":<34} {test_sonnet(test_path=GOLD, gold_path=GOLD):6.2f}')
  print(f'{"gold, tail capped at 128":<34} '
        f'{gold_cut_at_generation_budget(gold, tokenizer):6.2f}')

  print('\n=== random-character floors ===')
  summarize('random chars only', random_chars_only(gold))
  summarize('prompt + random tail', prompt_plus_random_tail(gold))
  summarize('prompt + random tail (capped)', prompt_plus_random_tail_capped(gold, tokenizer))

  print('\n=== floors, uncapped ===')
  print(f'{"prompt only (3 lines)":<34} {prompt_only(gold):6.2f}')
  summarize('shuffled gold', shuffled_gold(gold))
  summarize('prompt + wrong tail', prompt_plus_wrong_tail(gold))

  print('\n=== floors, capped at 128 new tokens ===')
  print(f'{"prompt only (cap non-binding)":<34} {prompt_only_capped(gold, tokenizer):6.2f}')
  summarize('shuffled gold (capped)', shuffled_gold_capped(gold, tokenizer))
  summarize('prompt + wrong tail (capped)', prompt_plus_wrong_tail_capped(gold, tokenizer))



