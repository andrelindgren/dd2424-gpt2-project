#
# PYTHONIOENCODING=utf-8 uv run python sonnet_generation.py --use_gpu --epochs 5 --batch_size 4 --lr 1e-5 --held_out_sonnet_path data/sonnets_held_out_dev.txt | tee output.txt
#

from evaluation import test_sonnet

score = test_sonnet(
    test_path='predictions/generated_sonnets_8.txt',
    gold_path='data/TRUE_sonnets_held_out_dev.txt'
)
print(f"CHRF score: {score}")