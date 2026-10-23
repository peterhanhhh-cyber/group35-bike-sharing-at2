# Bike-Sharing Demand Estimation — AT2 Group 35

41040 Introduction to Artificial Intelligence, University of Technology Sydney.

This project estimates the number of bike rentals in a given hour from calendar
and weather information, using the UCI Bike Sharing hourly dataset (Washington,
D.C., 2011–2012). Four techniques were developed under one shared protocol and
compared on the same frozen test rows.

**Group 35:** Ruo Wu, Yichen Xie, Junzhe Xu, Zongliang Han

## Results

| Method | Lab | Val MAE | Test MAE | Test R² |
|---|---|---:|---:|---:|
| **LSTM (selected)** | Lab 4 | **33.20** | **30.46** | **0.950** |
| 2D CNN | Lab 6 | 33.51 | 31.56 | 0.950 |
| Random Forest | Lab 3 | 40.23 | 35.56 | 0.925 |
| Autoencoder + head | Lab 8 | 54.96 | 58.47 | 0.840 |

The LSTM and the 2D CNN were only 0.31 rentals/hour apart on validation, so the
pre-registered weighted scorecard decided the selection. The autoencoder is
reported as a documented negative result.

## Notebooks

| # | Notebook | Open in Colab |
|---|---|---|
| 00 | [EDA and shared preparation](project/notebooks/00_eda_and_data_preparation.ipynb) | [Colab](https://colab.research.google.com/github/peterhanhhh-cyber/group35-bike-sharing-at2/blob/main/project/notebooks/00_eda_and_data_preparation.ipynb) |
| 01 | [Random forest](project/notebooks/01_random_forest_richard.ipynb) | [Colab](https://colab.research.google.com/github/peterhanhhh-cyber/group35-bike-sharing-at2/blob/main/project/notebooks/01_random_forest_richard.ipynb) |
| 02 | [LSTM](project/notebooks/02_lstm_jimmy.ipynb) | [Colab](https://colab.research.google.com/github/peterhanhhh-cyber/group35-bike-sharing-at2/blob/main/project/notebooks/02_lstm_jimmy.ipynb) |
| 03 | [2D CNN](project/notebooks/03_cnn_junzhe.ipynb) | [Colab](https://colab.research.google.com/github/peterhanhhh-cyber/group35-bike-sharing-at2/blob/main/project/notebooks/03_cnn_junzhe.ipynb) |
| 04 | [Autoencoder](project/notebooks/04_autoencoder_peter.ipynb) | [Colab](https://colab.research.google.com/github/peterhanhhh-cyber/group35-bike-sharing-at2/blob/main/project/notebooks/04_autoencoder_peter.ipynb) |
| **05** | **[Comparison and selection (best)](project/notebooks/05_comparison_and_selection.ipynb)** | **[Colab](https://colab.research.google.com/github/peterhanhhh-cyber/group35-bike-sharing-at2/blob/main/project/notebooks/05_comparison_and_selection.ipynb)** |

Notebooks 01–04 share one seven-section layout (setup, model, tuning experiment,
final model, results, cost and saved evidence, conclusion); the shared training
loop, evaluation and plots live in `project/src/bike_common.py`.

All notebooks are committed with their executed outputs. The Colab links are
for viewing; running them needs `hour.csv` and the `src/` folder in the Colab
working directory.

## Run the interface locally

```bash
git clone https://github.com/peterhanhhh-cyber/group35-bike-sharing-at2.git
cd group35-bike-sharing-at2
pip install -r requirements.txt
streamlit run project/ui/app.py
```

### Live demo

The app is a Streamlit application, so GitHub alone cannot execute it. To put it
online for free, use Streamlit Community Cloud and point it at this repository:

* Repository: `peterhanhhh-cyber/group35-bike-sharing-at2`
* Branch: `main`
* Main file path: `project/ui/app.py`

Deploy link:
https://share.streamlit.io/deploy?repository=https://github.com/peterhanhhh-cyber/group35-bike-sharing-at2&branch=main&mainModule=project/ui/app.py

The app loads `results/comparison_summary.json`, rebuilds the same 23 features
used in training, and calls the selected model. The demo scenario (June, 08:00,
Monday, humidity 0.70) returns about 624 rentals/hour.

## Repository layout

```
hour.csv, split_manifest.csv     raw data and frozen chronological split
project/src/bike_common.py       shared pipeline: split, features, metrics, training loop, plots
project/notebooks/               six executed notebooks
project/models/                  trained artifacts (RF stored compressed)
project/results/                 metrics JSON, predictions, comparison summary
project/figures/                 figures used in the notebooks and report
project/ui/app.py                Streamlit interface
```

## Data

Fanaee-T, H. and Gama, J. (2013) 'Event labeling combining ensemble detectors
and background knowledge', *Progress in Artificial Intelligence*, 2(2–3),
pp. 113–127. Dataset: UCI Machine Learning Repository, Bike Sharing Dataset.

