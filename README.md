<!-- README.md is generated from README.Rmd. Please edit that file -->

# synthsensor

<!-- badges: start -->

[![R-CMD-check](https://github.com/AstridMarie2/synthsensor/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/AstridMarie2/synthsensor/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

**This repo is modified from the original synthsensor package by Astrid Marie Skålvik
(@AstridMarie2) to generate anomalies for multivarirate time series.**

Original repository: [synthsensor GitHub](https://github.com/AstridMarie2/synthsensor.git)

Demo website: [synthsensor Interactive App](https://sensordiagnostics.shinyapps.io/app_synth/)

New modified code is located in folder `Python` and the original code is in folder `R`.

---


**synthsensor** generates **labeled, two-sensor synthetic time series**
for benchmarking sensor diagnostics and uncertainty methods. It supports
configurable background signals (AR(1), random walk, Poisson moving
average, sine), **delay/attenuation**, **noise & bias, spikes**
(correlated/uncorrelated), and **drifts**, and outputs per-timestep
**anomaly flags** and **true error** proxies.

- Reproducible datasets for detection/diagnostics research
- Scriptable functions **and** an interactive Shiny app
- Labeled outputs for supervised and rule-based evaluation
