# AITemperaturePrediction

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-neural%20network-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-archived-lightgrey)

Neural network post-processing of the **WRF** numerical weather model to forecast 2 m air temperature at weather stations across **Galicia (Spain)**. This project was carried out during my internship at **MeteoGalicia**, the Galician meteorological agency, with all data processing and training run on the **CESGA** supercomputing cluster.

> **About this repository:** this is a showcase of the work done during the internship. The original datasets (MeteoGalicia station observations and WRF model outputs) are no longer available, so the code **cannot be re-run**. The notebooks keep the outputs from the original executions, and the figures in `images/` are the actual results obtained at the time.

---

## The problem

Numerical weather models like WRF simulate the atmosphere on a grid, and their raw 2 m temperature forecasts carry systematic errors at specific locations: coastal effects, complex terrain, and particularly **daily maximum and minimum temperatures**. The goal was to learn a correction on top of WRF: given the model's forecast variables at a station, predict the temperature that was actually observed.

---

## Pipeline

### 1. Building the dataset

- Merged hourly **WRF forecasts** with hourly **observations from weather stations** across Galicia (2008–2025), aligned by station and timestamp.
- Model variables at several vertical levels: surface pressure, 2 m temperature, geopotential height, water vapour, wind components (U, V), temperature aloft, and precipitation-related fields.

### 2. Feature engineering

Galicia's climate changes a lot over short distances, driven by the Atlantic and a rugged terrain. To capture this, I added:

- **Sea percentage** around each station: a land/sea mask from MeteoGalicia's 1 km WRF grid, smoothed over a ~25 km window and assigned to each station by nearest-neighbour lookup (KD-tree).
- **Orography**, latitude and longitude.
- **Cyclical time features**: sine/cosine encodings of hour of day and day of year, so the model sees 23:00 → 00:00 and 31 Dec → 1 Jan as continuous.

![Sea mask, sea percentage and orography of Galicia](images/meteorological_variables.png)

### 3. Multicollinearity analysis

The same variable at nearby vertical levels is almost perfectly correlated (|r| > 0.9), which made a linear regression unstable, with huge coefficients of opposite sign cancelling each other out. Keeping a single representative level per variable group gave virtually the same R² (≈ 0.75) with interpretable coefficients. For the neural network, a new dataset was built using levels far apart in the vertical (0, 7 and 12) to reduce redundancy.

![Correlation matrix (only |r| > 0.7 shown)](images/correlation_matrix.png)

### 4. Linear baseline

A linear regression lowers the overall hourly MAE compared to raw WRF (**2.35 °C vs 3.47 °C**), but that result is misleading. Most hours are "ordinary" temperatures that are easy to fit, and the linear model **fails on daily extremes**, which are what matters most operationally:

| Mean absolute error of daily extremes | Linear model | WRF |
|---|---|---|
| Daily maximum | 2.78 °C | 1.43 °C |
| Daily minimum | 1.84 °C | 1.24 °C |

![Monthly error of the linear model vs WRF](images/linear_WRF_comparison.png)

This showed that a more flexible, non-linear model was needed.

### 5. Neural network (PyTorch)

A fully connected network trained on standardised inputs to predict the observed temperature:

- 3 hidden layers (128 → 64 → 32) with ReLU and dropout (0.2)
- Adam optimiser (lr = 1e-3, weight decay 1e-5), MSE loss, batch size 256
- `ReduceLROnPlateau` learning-rate scheduler and custom early stopping
- 80/20 train/validation split, keeping the best model by validation loss

![Neural network definition](images/neuralnet_class.png)

---

## Results

The neural network reduces the error of **both daily maximum and minimum temperatures** compared to WRF: medians of roughly 1.2 °C vs 1.6 °C for maxima and 1.1 °C vs 1.55 °C for minima, with a noticeably narrower error spread and fewer large misses.

![Neural network vs WRF on extreme temperatures](images/neuralnet_WRF_boxplot.png)

---

## Limitations

Some things I would do differently today:

- **Random train/validation split.** With hourly time series, neighbouring hours end up in both sets, which makes validation optimistic. A split by time period (e.g. train on earlier years, validate on the last one) would give a more honest estimate.
- **No independent test set.** Model selection and final evaluation used the same validation data.
- **Uniform loss weighting.** The dataset included a per-sample weight column that was never used in the loss. Weighting samples to emphasise extreme temperatures is the natural next step.
- **Not reproducible** since the data is gone (see note above).

---

## Repository structure

| Path | Contents |
|---|---|
| `NOTEBOOKS/data_preproccessing_github.ipynb` | Dataset construction, feature engineering, correlation analysis and linear baseline (with original outputs) |
| `NOTEBOOKS/Neural_network.ipynb` | Neural network design, training loop and evaluation, explained step by step |
| `images/` | Figures from the original runs |
| `requirements.txt` | Python packages used in the project |

---

## Tech stack

Python · PyTorch · scikit-learn · pandas · NumPy · xarray / netCDF · SciPy · Matplotlib / Seaborn · HPC (CESGA)

---

## License

Released under the [MIT License](LICENSE).
