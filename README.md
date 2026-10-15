# Predicting Jet Engine Remaining Useful Life (RUL) Using Machine Learning

## Overview
This project predicts the Remaining Useful Life (RUL) of a jet engine from its sensor readings. RUL is the number of flight cycles an engine has left before it fails. Knowing it early lets airlines repair engines before they fail, which avoids unexpected groundings and costly delays.

## Dataset
- **Source:** [Kaggle: NASA Turbofan Jet Engine Dataset CSV](https://www.kaggle.com/datasets/azmaininqiadadeeb/nasa-turbofan-jet-engine-dataset-csv), based on NASA's C-MAPSS simulation (FD001 subset)
- **Size:** 20,631 rows x 27 columns (training file)
- **Contents:** 100 engines, 21 sensors, 3 operating settings, 1 target (RUL)
- **Note:** The data is simulated run-to-failure data. Each engine runs until it fails.

## Workflow
1. **Data cleaning:** no missing values, no duplicates, correct data types.
2. **EDA:** descriptive statistics, histograms of every column, sensor trends over an engine's life, and correlation with RUL.
3. **Feature engineering:**
   - Dropped 8 columns that never change.
   - Created `temperature_difference` (HPC outlet temperature minus LPC outlet temperature).
   - Created a 3-cycle rolling average of HPC outlet temperature.
   - Removed 200 rows left empty by the rolling average (first 2 cycles of each engine).
4. **Modeling:**
   - 5 inputs: `cycle`, LPC outlet temperature, HPC outlet temperature, `temperature_difference`, rolling average of HPC outlet temperature.
   - Target: RUL.
   - Random 80/20 train/test split (16,344 train rows, 4,087 test rows).
   - Features scaled with StandardScaler, fitted on training data only.
   - Started with Linear Regression, then built a Random Forest Regressor (100 trees).
5. **Evaluation:** MAE, RMSE and R² score.

## Results

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---|---|---|
| Linear Regression | 31.21 | 40.64 | 0.63 |
| Random Forest | 28.29 | 38.71 | 0.67 |

- **MAE:** average miss, in cycles.
- **RMSE:** like MAE, but big misses count more.
- **R²:** 1 is perfect, 0 is no better than guessing the average.

  ## Key Insights
- **Engines live very different lengths.** Lifetimes range from 128 to 362 cycles (average about 206), so the cycle count alone can't tell how much life is left. The model has to read the sensors.
- **Sensors drift as the engine wears out.** Temperatures and HPC static pressure rise, while HPC outlet pressure falls. The drift is gradual and speeds up near failure.
- **8 columns never change** (or barely change) and were dropped because they can't help the model.
- **Smoothing helped.** The 3-cycle rolling average of HPC temperature correlated more strongly with RUL (-0.77) than the raw reading (-0.66).
- **`temperature_difference` did not beat its parts** in correlation (-0.63, against -0.68 and -0.66). Its effect on the model score was not tested separately.
- **Random Forest beat Linear Regression** (MAE 28.29 vs 31.21 cycles), but by a small margin.

## Interactive Graph
Pick an engine from the dropdown to compare its true RUL (blue) with the Random Forest prediction (dashed).

| Engine 1 | Engine 100 |
|---|---|
| 

![Engine 1](images/interactive_graph_engine_1.png)

 | 

![Engine 100](images/interactive_graph_engine_100.png)

 |

*The dropdown works when the notebook is running. These are screenshots. More engines are in the `images` folder.*

## Limitations
- **Scores are optimistic.** The test set is a random 20% of rows, so rows from the same engine appear in both training and test data. Splitting by whole engines would give a more honest score.
- The official test file (`test_FD001`) was not used.
- The data is simulated, not from real aircraft.
- Trained and tested on FD001 only (one operating condition, one fault type). Results may not carry over to FD002 to FD004.

## Project Structure
```
jet-engine-rul-prediction/
├── Real - Time Aircraft Engine Failure Prediction.ipynb   # Jupyter notebook
├── images/                                                # Graph screenshots
└── README.md                                              # Documentation
```
The dataset is not included. Download it from the Kaggle link above.

## How to Run
1. Download `train_FD001_descriptive.csv` from the Kaggle link and put it in the same folder as the notebook.
2. Install the libraries:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn ipywidgets
   ```
3. Open the notebook and click **Run All**.

## Tech Stack
- **Language:** Python
- **Data processing and feature engineering:** Pandas, NumPy
- **Machine learning:** Scikit-learn (Linear Regression, Random Forest Regressor)
- **Visualization and interactive graph:** Matplotlib, ipywidgets

## Author
**Mubarak Agboola**

Mechanical Engineering | Machine Learning and Data Analytics
