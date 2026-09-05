# CS 224N Default Final Project: Build GPT-2

Modified from CS 224N Default Final Project: Build GPT-2.

This project has followed some of the instructions in the CS 224N handout and implemented some important components of the GPT-2 model to better understand its architecture.

## Setup instructions

`uv sync`

## Part 1: Implemented the following parts:

The implemented/modified parts are in:
* `modules/attention.py`
* `modules/gpt2_layer.py` 
* `models/gpt2.py`
* `classifier.py`
* `optimizer.py`
* `sonnet_generation.py`

Sanity tests:
* `check_env.py`: To make sure the environment is working correctly.
* `sanity_check.py`: To test the implementation of GPT models.
* `optimizer_test.py`: To test the implementation of `optimizer.py`.
* `adamw_bench.py`: To see if one can speed up `optimizer.py`.


## Part 2.1: SST + CFMIDB classifier
Relevant files and folders:
* `classifier.py`

I completed the code in `classifier.py` in order to be able to train a classifier on the sst and cfimdb datasets. 
It now consists of a gpt + a linear head.

I modified it to include the ability to train either:
* last-linear-layer: the GPT parameters are frozen and the task specific head parameters are updated
* full-model: head + GPT parameters are updated as well
* embedding-only: head + word+pos embeddings are also trained 
* top-layers: head + last --num_unfrozen_layers transformer layers are also trained

For example, to train sst on a linear head:
`uv run python classifier.py --dataset sst --fine-tune-mode last-linear-layer --use_gpu --epochs 10 --batch_size 8 --lr 1e-3 --run_name sst-last-linear`


## Part 2.2 faster head training
Relevant files and folders:
* `extract_features.py`
* `/features/*.pt`
* `train_head.py`


In order to massively speed up the head hyperparameter search I extracted and stored the GPT-2 last-token features 
for sst and cfimdb for one whole epoch. I did this Because the GPT layers are frozen for only head training so the whole pass through the gpt 
only need to be run once for each training example.

Extract using: `extract_features.py`

Train using: `train_head.py`

I stored the extracted features in `/features/*.pt`


## Part 2.3 SST + CFMIDB classifier hyperparameter search
Relevant files and folders:
* `sweep.py`
* `analysis.py`
* `sst_analysis.txt`
* `results/`
* `results/runs.csv`
* `results/anytime_sst.png`

To run the hyperparameter search run the following:
`uv run python sweep.py --stage all --use_gpu`

To analyze the results from the search:
`uv run python analysis.py`

My results are stored in 
* `sst_analysis.txt`
* `results/`
* `results/runs.csv`
* `results/anytime_sst.png`

A graph showing the cumulative time taken to do hyperparameter search and the best dev acc found so far can be seen here:
`/results/anytime_sst.png`



## Part 3.1: Sonnet generation

Relevant files and folders:
* `sonnet_generation.py`
* `sonnet_evaluation.py`
* `sonnet_training_logs/`

I completed the code in `sonnet_generation.py` in order to be able to generate sonnets.

For example, to train:
`PYTHONIOENCODING=utf-8 uv run python sonnet_generation.py --use_gpu --epochs 5 --batch_size 4 --lr 1e-5 --held_out_sonnet_path data/sonnets_held_out_dev.txt | tee output.txt`

To evaluate it using CHRF score: `sonnet_evaluation.py`

## Part 3.2: Baselines, evaluation using CHRF score
Relevant files and folders:
* `sonnet_baseline.txt`
* `sonnet_baselines.py`
* `sonnet_baselines/`

### Part 3.2.1: Understanding
To understand the score better I created investigated different baselines for the sonnets.
I did this because of the concern that so many models had very similar scores but still far from a perfect `100`.
My results can be found in `sonnet_baseline.txt`.

### Part 3.2.2: Perfect (identity)
To make sure the evaluation could reach `` in score if it was a perfect match. And yes it can reach `100`.

### Part 3.2.3: Random
By just outputting random characters the model can reach `7.82`. A weak baseline.

### Part 3.2.4: Prompt + random tail
The evaluation includes and scores the 3 lines of gold seed given to the model. 
So the absolute floor if it doesnt generate anything else is already `24.77`.

If one includes a random tail after the prompt the score is `26.75`. 
This we can consider to be our starter baseline. Anything above this and the model has learn something.

### Part 3.2.5: Prompt + gold tail
By taking the prompt and adding a tail from another sonnet gold we can crate a baseline that one could consider be a style baseline.

If a model reaches this baseline of `41.84` we can say that has learnt what sonnet writing is 
and that it is writing in the style of the sonnets, using proper sentencing, structure, vocabulary etc.

### Part 3.2.6: Content
To go beyond `41.84` would require the model to learn to condition on the 3 seed lines to write a sonnet that
both: 1) have the same content as the sonnet in order to 2) also use the same words as the gold to express that content.

A hard problem, which we will see would not be solved by this model.

### Part 3.2.7: Uncapped vs capped
It was discovered late in training that all generations had a max length of 128 tokens which made it impossible for the 
models to actually write the sonnet to the correct length. 
I investigated this by capping the evaluation to `len(prompt) + 128 tokens` but the score difference was minimal.

All the evaluations in the sweep used the uncapped evaluation values.


## Part 3.3 Sonnet hyperparameter search
Relevant files and folders:
* `sonnet_runs.txt`
* `sonnet_sweep.py`
* `predictions/`
* `results_sonnet/sonnet_sweep.csv`
* `results_sonnet_seeded/sonnet_sweep_seeded.csv`
* `sonnet_sweep_seeded.py`
* `sonnet_sweep_seeded_results.txt`
* `sonnet_training_logs/`

First one has to train sonnets using `sonnet_generation.py`.

My commands are in `sonnet_runs.txt` and training logs in `sonnet_training_logs/`.

They can then be added to the list in `sonnet_sweep.py`

To run the hyperparameter search run: `uv run python sonnet_sweep.py`

My results can be seen in `results_sonnet/sonnet_sweep.csv`

To run a more narrow but seeded hyperparameter search run: `uv run python sonnet_sweep_seeded.py run`

My results can be seen in `results_sonnet_seeded/sonnet_sweep_seeded.csv`

To analyse the results one can run `uv run python sonnet_sweep_seeded.py analyze`

My generated analysis is saved here `sonnet_sweep_seeded_results.txt`

### Part 3.3.1 Hyperparameters
Learning rate and epochs are handled manually by training a model for each.

The sweeps cover temperature and top_p.

### Part 3.3.2 Initial search
The `sonnet_sweep.py` does a sweep through `checkpoints x temperature x top_p`.

The time requirement can quickly become large with many runs. 
This is because even thought the models only need to generate and not train, 
the generation is really inefficient and slow.

I did not search trhough all combinations.

### Part 3.3.3 Narrower seeded search
When the results started to come back from the initial sweep it was clear some configurations were better then others.
But also a problem arose. Becasue there are so many configs, 
the possibility that config was just lucky and not actually better became a problem.

So the more narrow seeded run uses only the most promising `temp` and `top_p` and do multiple seeds for each config 
 to get some statistical knowledge.


### Part 3.3.4 Resutls
For my experiments i got multiple equally good models that tied with first place.
```
checkpoint                temp  top_p     mean    sem   delta     z
5-0.0001-sonnet.pt         1.2    0.9   42.090  0.236   0.000  0.00  tied with best
5-3e-05-sonnet.pt          1.2    0.9   42.022  0.183   0.068  0.23  tied with best
10-0.0001-sonnet.pt        1.1   0.95   41.845  0.204   0.245  0.78  tied with best
5-0.0003-sonnet.pt         1.1    0.9   41.790  0.217   0.300  0.93  tied with best
10-0.001-sonnet.pt           2    0.9   41.583  0.103   0.508  1.97
10-0.0003-sonnet.pt        1.1      1   41.481  0.114   0.609  2.32
5-1e-05-sonnet.pt            1    0.9   41.465  0.269   0.625  1.75  tied with best
5-0.003-sonnet.pt          1.2    0.9   41.296  0.130   0.795  2.95
5-0.001-sonnet.pt          1.1   0.95   41.233  0.107   0.857  3.31
5-3e-06-sonnet.pt          1.2    0.9   38.771  0.705   3.319  4.47
```

And they also are in the ballpark of our style baseline of `41.84 +/- 0.29`.

So we can say that these winning models are equally good at generating text in the style of the sonnets. Hurray!

But to be able to say if it they also have learnt to write relevant content would require another benchmark and/or qualitative review. So that will have to be for future work.




