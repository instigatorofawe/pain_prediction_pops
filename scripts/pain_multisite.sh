#!/bin/bash
cd "$(dirname "$0")"
cd ..

python pops/ml/experiments/pain_score_multisite.py -c pain_pooled -o pain_multisite_all
python pops/ml/experiments/pain_score_multisite.py -c pain_pooled_mgh -o pain_multisite_mgh -l MGH
python pops/ml/experiments/pain_score_multisite.py -c pain_pooled_nwh -o pain_multisite_nwh -l NWH
python pops/ml/experiments/pain_score_multisite.py -c pain_pooled_bwh -o pain_multisite_bwh -l BWH
python pops/ml/experiments/pain_score_multisite.py -c pain_pooled_nsmc -o pain_multisite_nsmc -l NSMC