# Reproduce the submission

Tested with Python 3.13. The run takes a few minutes.

## A. Regenerate the submission file (uses the stored models)

1. Install the requirements.

   ```bash
   pip install -r requirements-solution.txt
   ```

2. Put the data bundle in `data/`, so that `data/tactical/` exists. Keep the file names.

   If the bundle is somewhere else, point to the folder that contains `tactical/`:

   ```bash
   export MTAH_DATA_DIR=/folder/that/contains/tactical
   ```

3. Run the pipeline.

   ```bash
   python solution/run_pipeline.py --out TikiTech.csv
   ```

4. Check the file.

   ```bash
   python evaluation/validate_submission.py TikiTech.csv
   ```

Same data and same code give the same file every time.

## B. Retrain the models (optional)

The models in `solution/models/` were made by the two scripts below. Run them to rebuild them
from the data. Every training label is made by an algorithm from the tracking; no hand labels are used.
Every random seed is fixed, so a rerun gives the same models.

1. Install the training requirements (adds PyTorch and hmmlearn).

   ```bash
   pip install -r requirements-train.txt
   ```

2. Put the data in place (step 2 above).

3. Train, in this order (the second one uses the first).

   ```bash
   python solution/train_dead_ball_ml.py
   python solution/train_possession_ml.py
   ```

4. Run part A again to write the submission file with the new models.

The training scripts **overwrite the files in `solution/models/`** and clear `solution/cache/`, so part A
picks up the new models with no other change. To get the shipped models back, run
`git checkout solution/models`.
