# PowerShell - Quick start for BlueTeamOps
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install numpy pandas scipy scikit-learn joblib
pip install lightgbm xgboost --upgrade --quiet --exists-action w || echo "Optional libs"
python .\ops\make_sample_data.py
python .\ops\freeze_baseline.py --data-dir .\PhishingData --out .\artifacts\frozen_baseline
python .\ops\audit_once.py --train .\PhishingData\train_data.csv --val .\PhishingData\val_data.csv --test .\PhishingData\test_data.csv --out .\artifacts\audit_report.json
python .\ops\groupcv_fpfn.py --train .\PhishingData\train_data.csv --val .\PhishingData\val_data.csv --folds 5 --outdir .\artifacts\groupcv
python .\ops\retrain_and_lock.py --train .\PhishingData\train_data.csv --val .\PhishingData\val_data.csv --test .\PhishingData\test_data.csv --target-fpr 0.003 --outdir .\artifacts
python .\ops\attack_harness.py --test .\PhishingData\test_data.csv --artifacts .\artifacts