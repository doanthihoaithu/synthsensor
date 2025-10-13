---
title: 'Synthetic Sensor Data Generation Framework'
tags:
  - synthetic data
  - time series
  - sensors
  - anomaly detection
  - uncertainty
  - R
  - Shiny
authors:
  - name: Astrid Marie Skålvik
    orcid: 0000-0002-0806-2818
    corresponding: true
    affiliation: 1 
affiliations:
 - name: University of Bergen, Department of Physics and Technology, Norway
   index: 1
date: 23 September 2025
bibliography: paper.bib

---


# Summary

Field datasets rarely include ground-truth labels for sensor errors, making it difficult to systematically develop and validate methods for anomaly detection, data quality control, and sensor diagnostics. 

This package provides an **R Shiny–based framework** for generating **synthetic time series from two correlated or redundant sensors**, with **user-configurable background signals** and **injected anomalies** (spikes, drifts, delay/attenuation, noise, and bias). The output includes both **anomaly labels** and **true error magnitudes**, enabling quantitative evaluation of diagnostic algorithms. 

The framework supports both interactive use through a Shiny app and programmatic use through R functions, making it useful for researchers and practitioners in sensor-driven domains. Its main goal is to facilitate **reproducible benchmarking** of sensor data quality assessment techniques in situations where controlled evaluation datasets are otherwise scarce.


# Statement of need
While real-world sensor datasets are valuable, they rarely include complete ground-truth labels for error types and magnitudes. This limits the ability to systematically evaluate diagnostic algorithms. 
<!-- To our knowledge, no existing R package offers synthetic generation of multi-sensor time series with flexible anomaly injection and labeled outputs. -->

This software addresses these gaps by:
- Allowing **two sensors** to measure either the same or correlated background signals.
- Providing a **broad set of configurable anomaly types**: spikes, drifts, sensor delay/attenuation, noise, and constant bias.
- Enabling **precise control over anomaly magnitude, duration, and correlation** between sensors.
- Outputting **CSV files with labeled anomalies** for use in supervised learning and method validation.
- Offering an **interactive Shiny UI** for rapid experimentation, as well as scriptable functions for batch generation.

# Related work
Several tools provide synthetic time-series data or anomaly-related functionality, but to our knownledge none combine **multi-sensor redundancy**, **configurable anomaly injection**, and **ground-truth error magnitudes**. Table 1 summarizes the most relevant approaches.

| Tool / Package | Key Features | Limitations vs. our framework |
|----------------|--------------|-------------------------------|
| **SimTensor (MATLAB)** [@simtensor2016]| Generates synthetic tensors with periodicity, seasonal patterns, and change-points. | No sensor redundancy; no labeled error magnitudes; not designed for anomaly benchmarking. |
| **GAN-based generators (e.g. TimeGAN, Python)**  [@yoon2019timegan] | Produces realistic synthetic time series via deep generative models. | Relies on training data. No explicit anomaly type, magnitude, or timing control; no labeled errors. |
| **Mutation-based anomaly frameworks (e.g. MDPI 2024)**  [@mutation2024]| Mutates multivariate series to introduce rare events. | Limited anomaly control; no error magnitudes; no multi-sensor correlation. |
| **R packages for anomaly detection (`anomalize`, `otsad`)** [@anomalize2018, otsad2019] | Detect anomalies in real or synthetic data. | Focus on detection, not generation; no ground truth labels or anomaly injection. |
| **MATLAB/Simulink examples**  [@matlab2025]| Demonstrate anomaly detection in streaming multichannel signals. | Simple signals; no flexible anomaly generation; no dataset export. |

*Table 1: Comparison of related tools with respect to anomaly generation and benchmarking functionality.*



# Use
The framework has been adopted in several research projects. It is used in [@Skalvik2025] to generate controlled datasets for validating a Bayesian method for error and uncertainty estimation of environmental sensor data. In [@Heggedal2025], it was applied in the development of a software tool that assists experts with annotating marine time series. In [@Stuen2025] it was used for demonstrating a method for uncertainty quantification of hydrogen leaks.

Beyond these projects, the package is broadly useful for:
- **Reproducible benchmarking**: researchers can share identical datasets (using fixed seeds) to compare diagnostic methods.
- **Teaching and training**: students can experiment with sensor error effects without requiring access to field instruments.
- **Rapid prototyping**: practitioners can quickly simulate “what if” scenarios for sensor anomalies before deploying to the field.


# Implementation
The framework is implemented in **R** and is provided both as:
1. A **Shiny application** for interactive configuration and dataset export.
2. A set of **modular R functions** for programmatic dataset generation.

## Background signal
The package provides functions to generate several background signal models:
- **Autoregressive process (AR(1))** — for stationary signals with tunable autocorrelation.
- **Random walk** — a special case of AR(1) with strong temporal autocorrelation.
- **Poisson moving average** — for smoothed stochastic signals.
- **Sine wave** — for periodic variation.

For the AR(1) and Random walk, the background signals can be **correlated** between sensors with user-defined cross-correlation. For the Poisson moving average and the Sine wave, the background signals are identical between the sensors. 

## Anomaly types
Anomalies can be introduced individually or in combination:
- **Delay/attenuation** — models exponential moving average sensor response.
- **Noise and constant bias** — configurable per sensor, with optional correlation between noise terms.
- **Spikes** — short-duration deviations, correlated or uncorrelated between sensors, with random sign and magnitude.
- **Drifts** — gradual deviations over time, with configurable slope and duration.

Anomalies are inserted at random positions while ensuring spacing from time-series edges to avoid boundary artifacts.

## Output
The output is a **CSV file** containing:
- Time series for both sensors.
- Flags for each time step indicating anomaly type.
- True error magnitude for each sensor.

# Reproducibility
All random processes are controlled via a user-specified random seed. Full mathematical details of signal generation and anomaly injection are provided in the software documentation and supplementary materials (https://github.com/AstridMarie2/synthsensor --> *Synthetic Sensor Data Generation Framework - Mathematics*). 

All functionality is covered by automated tests (testthat) and continuous integration (GitHub Actions).

# Example
An example workflow for generating a dataset with correlated AR(1) background signal, additive Gaussian noise, and occasional spikes:

```r
library(synthsensor)

df <- generate_data_function(
  n = 300,
  add_background   = TRUE,
  background_type  = "Sine Wave",
  sine_amplitude   = 2,
  sine_period      = 30,
  sd1 = 0.15, sd2 = 0.15,
  crosscor_noise   = 0,
  mean1 = 0.5, mean2 = -0.5,
  # anomalies
  n_spikes_corr    = 0,
  n_spikes_s1      = 3,
  n_spikes_s2      = 2,
  max_spike_length = 6,
  n_drifts_s1      = 1,
  n_drifts_s2      = 0,
  drift_duration   = c(25, 35),
  drift_slope      = c(0.05, 0.08),
  delayed_sensor   = "None",
  alpha_ema        = 0.3
)

head(df)

#>   TimeSinceClean  Sensor1    Sensor2                Date AnomalyFlag1 AnomalyFlag2 Measurand1  Measurand2      Diff
#> 1              1 1.024476 -0.1734142 2025-01-01 00:00:00       Normal       Normal  0.9158234 -0.08417662 1.1978903
#> 2              2 1.427743  0.2737805 2025-01-01 01:00:00       Normal       Normal  1.3134733  0.31347329 1.1539626
#> 3              3 1.817718  0.9042105 2025-01-01 02:00:00       Normal       Normal  1.6755705  0.67557050 0.9135072
#> 4              4 2.145533  0.9916997 2025-01-01 03:00:00       Normal       Normal  1.9862897  0.98628965 1.1538333
#> 5              5 2.298991  1.2462778 2025-01-01 04:00:00       Normal       Normal  2.2320508  1.23205081 1.0527134
#> 6              6 2.353803  1.6542066 2025-01-01 05:00:00       Normal       Normal  2.4021130  1.40211303 0.6995960


write.csv(df, "synthetic_dataset.csv", row.names = FALSE)
```

Alternatively, launch the Shiny app:
```r
synthsensor::app_synth()
```
# Dependencies

* R (>= 4.0)
* shiny
* dplyr
* ggplot2
* MASS
* stats
* gridExtra
* zoo

# Availability
The framework is available as:
1. An online Shiny app: https://sensordiagnostics.shinyapps.io/app_synth/
2. Source code and documentation on GitHub: https://github.com/AstridMarie2/synthsensor

Zenodo DOI link: https://doi.org/10.5281/zenodo.17157665

# Acknowledgements
Development of this framework was motivated by the need for labeled multi-sensor datasets for Bayesian error modeling in oceanographic instrumentation. 

# Funding
This work is part of the SFI Smart Ocean (a Centre for Research-based Innovation). The Centre is funded by the partners in the Centre and the Research Council of Norway (project no. 309612).

# References
