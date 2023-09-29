#!/bin/bash
cd "$(dirname "$0")"
cd ..

python pops/ml/experiments/pain_score_bootstrap.py -c all -n 200  -a
python pops/ml/experiments/pain_score_bootstrap.py -c mgh -n 200 -l MGH -a
python pops/ml/experiments/pain_score_bootstrap.py -c bwh -n 200 -l BWH -a
python pops/ml/experiments/pain_score_bootstrap.py -c nsmc -n 200 -l NSMC -a
python pops/ml/experiments/pain_score_bootstrap.py -c nwh -n 200 -l NWH -a