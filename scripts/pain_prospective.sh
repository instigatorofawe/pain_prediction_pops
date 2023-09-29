#!/bin/bash
cd "$(dirname "$0")"
cd ..

python pops/ml/experiments/pain_score_prospective.py -c pain_pooled/epoch=2-step=4674.ckpt -o pain_prospective_all
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_binarized_4/epoch=1-step=3116.ckpt -o pain_prospective_all_binarized_4
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_binarized_6/epoch=1-step=3116.ckpt -o pain_prospective_all_binarized_6
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_mgh/epoch=3-step=2720.ckpt -o pain_prospective_mgh
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_mgh_binarized_4/epoch=2-step=2040.ckpt -o pain_prospective_mgh_binarized_4
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_mgh_binarized_6/epoch=1-step=1360.ckpt -o pain_prospective_mgh_binarized_6
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_nwh/epoch=1-step=398.ckpt -o pain_prospective_nwh
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_nwh_binarized_4/epoch=2-step=597.ckpt -o pain_prospective_nwh_binarized_4
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_nwh_binarized_6/epoch=1-step=398.ckpt -o pain_prospective_nwh_binarized_6
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_bwh/epoch=1-step=1164.ckpt -o pain_prospective_bwh
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_bwh_binarized_4/epoch=1-step=1164.ckpt -o pain_prospective_bwh_binarized_4
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_bwh_binarized_6/epoch=0-step=582.ckpt -o pain_prospective_bwh_binarized_6
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_nsmc/epoch=4-step=495.ckpt -o pain_prospective_nsmc
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_nsmc_binarized_4/epoch=2-step=297.ckpt -o pain_prospective_nsmc_binarized_4
python pops/ml/experiments/pain_score_prospective.py -c pain_pooled_nsmc_binarized_6/epoch=1-step=198.ckpt -o pain_prospective_nsmc_binarized_6