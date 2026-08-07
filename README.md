# Overview

This repository contains the Python analysis scripts and the model performance results for the following paper:

Investigating Indoor CO₂ Heterogeneity: The Impact of Vertical Stratification on IAQ Assessment and Presence Detection in Brazilian Educational Buildings

# Repository Contents


```
├──── 0_results/                        # model performance results
│ ├──── building_L/
│ │ ├─── analysis_sensor_position/      # single sensor scenario
│ │ │ ├── _timeindep_results_raw/       # ensemble model
│ │ │ └── _timeseries_results_raw/      # LSTM
│ │ │
│ │ └─── analysis_vertical_difference/  # dual sensor scenario
│ │   ├── _timeindep_results_raw/       # ensemble model
│ │   └── _timeseries_results_raw/      # LSTM
│ │
│ └──── building_S/
│   ├─── analysis_sensor_position/      # single sensor scenario
│   │ ├── _timeindep_results_raw/       # ensemble model
│   │ └── _timeseries_results_raw/      # LSTM
│   │
│   └─── analysis_vertical_difference/  # dual sensor scenario
│     ├── _timeindep_results_raw/       # ensemble model
│     └── _timeseries_results_raw/      # LSTM
│    
│
├──── 1_analysis/                       # statistical analysis results
│ ├──── building_L/
│ │ ├─── analysis_sensor_position/      # single sensor scenario
│ │ └─── analysis_vertical_difference/  # dual sensor scenario
│ │
│ └──── building_S/
│   ├─── analysis_sensor_position/      # single sensor scenario
│   └─── analysis_vertical_difference/  # dual sensor scenario
│
│
├──── requirements.txt                  # python env requirements
│
└──── README.md
```

# Examples of Use

## Configure Python environment

We are using Python 3.11, but it should also run on version 3.12 or higher (not tested).

Set up the environment using `pip` with `requirements.txt`, we recommend using a virtual environment:

```
pip install -r requirements.txt
```

> Note that PyTorch is installed as the CPU version by default through `pip`. If you need GPU acceleration, please manually install the PyTorch version corresponding to your GPU's CUDA version. We are using `2.7.0+cu128`.


## Minimalist example of model training code

> Note: Since the original training Jupyter Notebooks contain raw data in the output that cannot be made public directly, we are providing only an equivalent minimalist example here. 

### Two-stage grid search

> Using the 5-fold CV for the base learner SVM in the ensemble model as an example

First round:

```python
best_param = {}

# SVC
# grid search using 5-fold cross-validation

start_time = datetime.now()

tuned_parameters = [
    {"kernel": ["rbf"],
     "gamma": [0, 2**(-14), 2**(-12), 2**(-10), 2**(-8), 2**(-6), 2**(-4), 2**(-2), 2**0, 2**2, 2**4, 2**6, 2**8, 2**10],
     "C": [2**(-10), 2**(-8), 2**(-6), 2**(-4), 2**(-2), 2**0, 2**2, 2**4, 2**6, 2**8, 2**10],
     "class_weight": ["balanced"]},
]

# 5-fold cross-validation grid search
grid_search = GridSearchCV(
    SVC(probability=True),
    tuned_parameters,
    cv=5,
    # scoring="balanced_accuracy",
    scoring="roc_auc",
    verbose=2
)
grid_search.fit(X_train, y_train)

end_time = datetime.now()
print("-> grid search done, total time used: ", end_time - start_time)
```

Second round:

```python
# refine the grid near best parameters
C_best = grid_search.best_params_["C"]
gamma_best = grid_search.best_params_["gamma"]

C_ls_ref = [C_best / 2, C_best, C_best * 2]
gamma_ls_ref = [gamma_best / 2, gamma_best, gamma_best * 2] if gamma_best != 0 else [gamma_best, 2**(-15), 2**(-14)]

# SVC
# grid search using 5-fold cross-validation

start_time = datetime.now()

tuned_parameters = [
    {"kernel": ["rbf"],
     "gamma": gamma_ls_ref,
     "C": C_ls_ref,
     "class_weight": ["balanced"]},
]

# 5-fold cross-validation grid search
grid_search = GridSearchCV(
    SVC(probability=True),
    tuned_parameters,
    cv=5,
    # scoring="balanced_accuracy",
    scoring="roc_auc",
    verbose=2
)
grid_search.fit(X_train, y_train)

end_time = datetime.now()
print("-> grid search done, total time used: ", end_time - start_time)
```

3-fold expanding-window time-series CV for LSTM:

```python
# Grid search
# 15 min / 30 min / 60 min / 90 min / 120 min / 150 min / 180 min
window_candidates = [3, 6, 12, 18, 24, 30, 36]
# batch size
batch_candidates = [32, 64]
# hidden layer units
units_candidates = [8, 16, 32, 64, 128, 256]
# number of layer
layer_candidates = [1, 2, 3]

grid_results = []

# ...

for window_size in window_candidates:
    # no test data for grid search
    X_seq, y_seq = create_lstm_dataset(X_train_full, y_train_full, window_size)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    for batch_size in batch_candidates:
        for lstm_units in units_candidates:
            for num_layers in layer_candidates:
                print(f"== window_size: {window_size}, batch_size: {batch_size}, lstm_units: {lstm_units}, layers: {num_layers} ==")

                tscv = TimeSeriesSplit(n_splits=3)

                accs = []
                aucs = []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
                    # split to train / validation data
                    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
                    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

                    train_ds = TensorDataset(X_train, y_train)
                    val_ds = TensorDataset(X_val, y_val)
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
                    val_loader = DataLoader(val_ds, batch_size=batch_size)

                    model = LSTMClassifier(X_train.shape[2], lstm_units, num_layers=num_layers)
                    criterion = nn.BCELoss()
                    optimizer = torch.optim.Adam(model.parameters())

                    model = train_model(model, train_loader, val_loader, criterion, optimizer, device)

                    model.eval()
                    with torch.no_grad():
                        val_probs = model(X_val.to(device)).cpu().numpy()
                        val_preds = (val_probs >= 0.5).astype(int)
                    acc = accuracy_score(y_val, val_preds)
                    auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else np.nan

                    accs.append(acc)
                    aucs.append(auc)
                    print(f"    Fold {fold+1}: Acc. = {acc:.4f} | AUC = {auc:.4f}")

                mean_acc = np.mean(accs)
                mean_auc = np.nanmean(aucs)
                print(f"window_size {window_size} / batch_size {batch_size} / lstm_units {lstm_units}: mean Acc. = {mean_acc:.4f}, mean AUC = {mean_auc:.4f}")
```

### Permutation importance analysis

For ensemble model:

```python
# ...
for random_seed in range(loop):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    # ...
    # permutation importance of features
    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=random_seed, n_jobs=-1
    )
    importances = perm.importances_mean
    feature_names = X.columns if hasattr(X, "columns") else [f"feature_{i}" for i in range(X.shape[1])]
    feature_importance = {
        name: score for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    }
    # ...
```

For LSTM:

```python
# permutation importance for LSTM
def lstm_permutation_importance(model, X, y, metric_fn, n_repeats=10, seed=42):
    np.random.seed(seed)
    model.eval()
    with torch.no_grad():
        base_pred = model(X).cpu().numpy()
    base_score = metric_fn(y.cpu().numpy(), (base_pred >= 0.5).astype(int))
    importances = []

    for feat_idx in range(X.shape[2]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.clone()
            for t in range(X.shape[1]):
                idx = torch.randperm(X.shape[0])
                X_perm[:, t, feat_idx] = X_perm[idx, t, feat_idx]
            with torch.no_grad():
                perm_pred = model(X_perm).cpu().numpy()
            score = metric_fn(y.cpu().numpy(), (perm_pred >= 0.5).astype(int))
            scores.append(score)
        mean_drop = base_score - np.mean(scores)
        importances.append(mean_drop)

    return importances
```

> All model performance results are stored as CSV files in `0_results`.

## Run statistical analysis

Under `1_analysis`, find the Jupyter Notebook file (`anova.ipynb`) corresponding to the building and sensor configuration. You can rerun the code line by line starting from the beginning.

