# Reproduce the submission

This loads the stored models in `solution/models/`. 

```bash
pip install -r requirements-solution.txt
export MTAH_DATA_DIR=/path/to/the/data/bundle
python solution/run_pipeline.py --out TikiTech.csv
python evaluation/validate_submission.py TikiTech.csv
```
