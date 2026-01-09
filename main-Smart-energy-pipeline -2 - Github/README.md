NEC Smart Energy Pipeline

ML pipeline for selecting the best power plant for different demand scenarios.

What This Does

The National Energy Consortium needs to pick which power plant to use each day from 64 options. Running simulations to figure out costs is expensive, so we built a machine learning model that predicts which plant will be cheapest for each demand scenario.

We use two models: Gradient Boosting (main one) and Random Forest (backup option).

Quick Start

Install dependencies:
pip install -r requirements.txt


Run the pipeline:
python main.py


Warning: This takes 1-2 hours because it runs cross-validation on 500 demand scenarios.

How It Works

Load CSVs → Clean Data → Train Model → Evaluate → Save Results


What happens:
1. Loads demand.csv, plants.csv, and generation_costs.csv
2. Fixes missing values and filters out plants that are never competitive
3. Creates sklearn Pipeline (preprocessing + model combined)
4. Runs Leave-One-Group-Out cross validation (tests each demand individually)
5. Saves everything to artifacts/ folder

Configuration

Change settings in config/config.yaml:

To switch models:
yaml
model:
  type: "random_forest"  # or "gradient_boosting"

To change test size:
yaml
evaluation:
  test_size: 20  # number of test demands

No code changes needed - just edit the yaml and rerun.

## What You Get

After running, check the artifacts folder:

- best_model.pkl - the trained pipeline (use this for predictions)
- preprocessor.pkl - just the preprocessing part
- performance_summary.json - all the metrics (RMSE, R²)
- selection_results.csv - shows which plant was picked for each demand
- cv_results.csv - RMSE for each cross-validation fold
- config_used.yaml - snapshot of the config you used

## Project Structure

smart-energy-pipeline-2/
├── config/
│   └── config.yaml           # settings
├── data/
│   ├── demand.csv
│   ├── plants.csv
│   └── generation_costs.csv
├── src/
│   ├── data_loader.py        # loads the CSVs
│   ├── preprocessing.py      # cleans data, filters plants
│   ├── model_trainer.py      # creates the sklearn pipeline
│   ├── evaluator.py          # custom scoring + cross validation
│   └── tuner.py              # hyperparameter search (disabled by default)
├── artifacts/                # created when you run main.py
├── main.py                   # run this
├── requirements.txt
└── README.md



How We Built This

This integrates work from our individual assessments:

Preprocessing approach:
- Replace "NA" with "NORAM" (otherwise Python thinks it's null)
- Fill missing values with median (137 missing in demand data, 96 in costs)
- Filter out 12 plants that never appear in top 10 cheapest

Model selection:
- Tested both Gradient Boosting and Random Forest in individual work
- Gradient Boosting performed best (3.42 RMSE vs 6.47 for RF)
- Used hyperparameters found through GridSearchCV in individual assignments

Evaluation:
- Custom error metric from the brief: Error(d) = min{c(p,d)} - c(p_ml,d)
- Leave-One-Group-Out CV (each demand tested on model that never saw it)
- Grouped train/test split to prevent data leakage

Using the Pipeline

For new data:

1. Put new CSVs in `data/` folder (same format as originals)
2. Run `python main.py`
3. Check `artifacts/selection_results.csv` for predictions

Understanding the outputs:

performance_summary.json has:
- Train/test RMSE
- LOGO CV mean and std
- R² score
- How many demands were predicted perfectly

selection_results.csv shows:
- Demand_ID
- Optimal_Cost (true best plant)
- Selected_Cost (what the model picked)
- Error (negative = model picked suboptimal plant)

Common issues:

"Could not find CSV files"
- Make sure files are named exactly: demand.csv, plants.csv, generation_costs.csv
- Check they're in the data/ folder

Takes forever to run
- LOGO CV on 500 folds takes time
- Use Random Forest if it is needed faster
- Or reduce test_size in config (but results will be less reliable)

Out of memory
- Try setting `n_jobs: 1` in config.yaml instead of -1
- This uses less RAM but takes longer

Technical Notes


We use a Pipeline so preprocessing and model are bundled together. When you save the model, it remembers how to preprocess new data.

Why these hyperparameters:
- n_estimators=150, learning_rate=0.2, max_depth=3
- Found these through grid search in individual work
- Tried ranges: estimators [50,100,150], learning_rate [0.05,0.1,0.2], depth [3,4,5]
- These gave best cross-validation scores

Data preprocessing:
- One-hot encode: region, day type, plant type (4 categorical features)
- Keep all 12 demand features (DF1-DF12) and 18 plant features (PF1-PF18)



