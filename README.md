# Overview

This repository contains the Python analysis scripts and the model performance results for the following paper:

Investigating Indoor CO₂ Heterogeneity: Impacts on IAQ Assessment and Data-Driven Presence Detection in Brazilian Educational Buildings

# Repository Contents


```
├──── 01_results/                       # model performance results
│ ├──── building_A/
│ │ ├─── analysis_sensor_position/      # single sensor scenario
│ │ │ ├── _timeindep_results_raw/       # ensemble model
│ │ │ └── _timeseries_results_raw/      # LSTM
│ │ │
│ │ └─── analysis_vertical_difference/  # dual sensor scenario
│ │   ├── _timeindep_results_raw/       # ensemble model
│ │   └── _timeseries_results_raw/      # LSTM
│ │
│ └──── building_B/
│   ├─── analysis_sensor_position/      # single sensor scenario
│   │ ├── _timeindep_results_raw/       # ensemble model
│   │ └── _timeseries_results_raw/      # LSTM
│   │
│   └─── analysis_vertical_difference/  # dual sensor scenario
│     ├── _timeindep_results_raw/       # ensemble model
│     └── _timeseries_results_raw/      # LSTM
│    
│
├──── 02_analysis/                      # statistical analysis results
│ ├──── building_A/
│ │ ├─── analysis_sensor_position/      # single sensor scenario
│ │ └─── analysis_vertical_difference/  # dual sensor scenario
│ │
│ └──── building_B/
│   ├─── analysis_sensor_position/      # single sensor scenario
│   └─── analysis_vertical_difference/  # dual sensor scenario
│
└──── README.md
```
