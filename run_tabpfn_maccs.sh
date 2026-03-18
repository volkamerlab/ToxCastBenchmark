#!/bin/bash
set -e

export PYTHONPATH=$PYTHONPATH:$(pwd)/models

pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    catboost \
    hyperopt \
    tabpfn==2.2.1 \
    tabpfn-extensions==0.1.6 \
    autogluon.tabular==1.4.0

python3 run_pipeline/robust_ToxCast_predicitons_and_HT_maccs.py
