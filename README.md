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
└──── README.md
```
