import itertools
import json
import random
import shutil
import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
from tqdm import tqdm
import os

import matplotlib.patches as mpatches

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from synthetic_generation import new_flag_vec, pick_spike_starts, update_flags, pick_segment, lagged_ema
from utils import set_random_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BACKGROUND_TYPE_MAP = {
    'ar_1_process': 'AR(1) Process',
    'random_walk': 'Random Walk',
    'sine_wave': 'Sine Wave',
    'poisson_moving_average': 'Poisson Moving Average'
}

class ANOMALY_TYPE(Enum):
    SPIKE = 1
    CORRELATED_SPIKE = 2
    DRIFT = 3
    CORRELATED_DRIFT = 4

BACKGROUND_TYPE_WITH_SUPPORTED_ANOMALY_TYPES = dict({
    'ar_1_process': [ANOMALY_TYPE.SPIKE, ANOMALY_TYPE.CORRELATED_SPIKE],
    'random_walk': [ANOMALY_TYPE.SPIKE, ANOMALY_TYPE.CORRELATED_SPIKE],
    'sine_wave': [ANOMALY_TYPE.SPIKE, ANOMALY_TYPE.CORRELATED_SPIKE, ANOMALY_TYPE.DRIFT, ANOMALY_TYPE.CORRELATED_DRIFT],
    'poisson_moving_average': [ANOMALY_TYPE.SPIKE, ANOMALY_TYPE.CORRELATED_SPIKE, ANOMALY_TYPE.DRIFT, ANOMALY_TYPE.CORRELATED_DRIFT]
})

BACKGROUND_TYPES_SUPPORTING_NORMAL_CORRELATION = ['ar_1_process', 'random_walk']

def generate_normal_data_for_random_walk_group(sensor_ids, correlated_cov):
    return None
def generate_normal_data_for_ar_1_process_group(sensor_ids, correlated_cov):
    return None

def generate_normal_data_for_sine_wave_group(sensor_ids, config):
    return None

def generate_normal_data_for_poisson_moving_average_group(sensor_ids, config):
    return None


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to preprocess the pivoted dataset, dividing into different feature subsets.

    Steps:
    1. Load and preprocess the data.
    2. Save processed data to disk.

    Parameters:
    - cfg: DictConfig, configuration object containing paths, parameters, and settings.
    """
    print(cfg)
    log.info("Starting main function")

    random_seed = cfg.random.seed
    set_random_seed(random_seed)

    used_settings = cfg.generation.used_settings

    config_dir = cfg.generation.config_dir
    for f in os.listdir(config_dir):
        settings_path = os.path.join(config_dir, f)
        if 'base' in f:
            continue
        if f.startswith(f'{used_settings}.yaml'):
            with open(settings_path, "r") as settings_file:
                settings_cfg = OmegaConf.load(settings_file)
                generation_cfg = cfg.generation
                dfs, training_df = generate_data_function_from_cfg(settings_cfg, generation_cfg)
                save_generated_data(dfs, training_df, settings_cfg, cfg)
def save_generated_data(testing_dfs, training_df, settings_cfg, global_cfg):
    data_dir = global_cfg.generation.data_dir
    saved_file_dir = os.path.join(data_dir, settings_cfg.settings_name)
    if os.path.exists(saved_file_dir):
        shutil.rmtree(saved_file_dir)
    os.makedirs(saved_file_dir, exist_ok=True)
    training_saved_file_path = os.path.join(saved_file_dir, 'synthetic_training.csv')
    training_df.to_csv(training_saved_file_path, index=False)
    log.info(f'Saved training data to {training_saved_file_path}')
    for index, df in enumerate(testing_dfs):
        saved_file_path = os.path.join(saved_file_dir, f'synthetic_{index}.csv')
        df.to_csv(saved_file_path, index=False)
        log.info(f'Saved generated data to {saved_file_path}')

        if global_cfg.generation.plot_data == True:
            fig = plot_generated_data(df, settings_cfg, global_cfg)
            saved_file_path = os.path.join(saved_file_dir, f'figure_synthetic_{index}.png')
            fig.savefig(saved_file_path, bbox_inches='tight')
            log.info(f'Saved figure to {saved_file_path}')

    if global_cfg.generation.plot_data == True:
        fig = plot_generated_data(training_df, settings_cfg, global_cfg)
        saved_file_path = os.path.join(saved_file_dir, f'figure_synthetic_training.png')
        fig.savefig(saved_file_path, bbox_inches='tight')
        log.info(f'Saved figure for training data to {saved_file_path}')

def plot_generated_data(df, settings_cfg, global_cfg):
    num_sensors = settings_cfg.num_sensors
    data_dir = global_cfg.generation.data_dir
    fig, axes = plt.subplots(nrows=num_sensors, ncols=1, figsize=(15, 10))
    for i in range(num_sensors):
        axes[i].plot(df[f'Sensor{i}'], label=f'Sensor{i}')
        axes[i].legend(loc="upper right")
        axes[i].set_xlim(0, df.index.max())

        sensor_anomalies = df[df[f'AnomalyFlag{i}'] != 'Normal']
        for index, anomaly in sensor_anomalies.iterrows():
            # print(index, anomaly[f'AnomalyFlag{i}'])
            color = 'red'
            if anomaly[f'AnomalyFlag{i}'] == 'Spike':
                color = 'red'
            elif anomaly[f'AnomalyFlag{i}'] == 'Drift':
                color = 'orange'
            elif anomaly[f'AnomalyFlag{i}'] == 'SpikeCorr':
                color = 'green'
            elif anomaly[f'AnomalyFlag{i}'] == 'DriftCorr':
                color = 'blue'
            elif anomaly[f'AnomalyFlag{i}'] == 'Both':
                color = 'violet'
            axes[i].axvline(index, color=color, alpha=0.5)

    red_patch = mpatches.Patch(color='red', label='Spike')
    orange_patch = mpatches.Patch(color='orange', label='Drift')
    green_patch = mpatches.Patch(color='green', label='SpikeCorr')
    blue_patch = mpatches.Patch(color='blue', label='DriftCorr')
    violet_patch = mpatches.Patch(color='violet', label='Both')
    #
    patches = [red_patch, orange_patch, green_patch, blue_patch, violet_patch]
    # patches = [red_patch]

    fig.suptitle(f'Sensor data generation', fontsize=16, y=1.08)
    # for patch in patches:
    #     axes[1].add_patch(patch)
    # axes[1].legend(loc="best")

    fig.legend(handles=patches, loc="lower center", ncol=len(patches), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return fig


def validate_settings_cfg(settings_cfg):
    log.info('Validating settings cfg: {}'.format(settings_cfg))
    num_sensors = settings_cfg.num_sensors
    background_types = ['sine_wave']*num_sensors
    # sensor_background_types = settings_cfg.sensor_background_types
    # for t in sensor_background_types:
    #     assert t in BACKGROUND_TYPE_MAP.keys()
    correlated_sensor_groups = settings_cfg.correlated_sensor_groups
    for key, value in correlated_sensor_groups.items():
        background_type = key[:-7]
        if background_type == 'random_walk':
            for group_id, group in value.groups.items():
                sensor_ids = group.sensor_ids
                for sensor_id in sensor_ids:
                    assert sensor_id < num_sensors
                    background_types[sensor_id] = background_type
        elif background_type == 'ar_1_process':
            for group_id, group in value.groups.items():
                sensor_ids = group.sensor_ids
                for sensor_id in sensor_ids:
                    assert sensor_id < num_sensors
                    background_types[sensor_id] = background_type
        elif background_type == 'independent':
            for group_id, group in value.items():
                sensor_id = group.sensor_id
                background_type = group.background_type
                assert sensor_id < num_sensors
                background_types[sensor_id] = background_type
    for key, val in settings_cfg.correlated_anomaly_groups.items():
        sensor_ids = val['sensor_ids']
        for sensor_id in sensor_ids:
            assert sensor_id <= num_sensors
    return background_types

def generate_training_data(settings_cfg, generation_cfg):
    num_data_point_per_batch = settings_cfg.num_data_point_per_batch
    n = max(int(num_data_point_per_batch), 100)
    num_sensors = settings_cfg.num_sensors
    num_training_points = settings_cfg.num_training_points

    poisson_k = generation_cfg.poisson_k
    poisson_lambda = generation_cfg.poisson_lambda
    randomwalk_sd = generation_cfg.randomwalk_sd
    background_rho_rw = generation_cfg.background_rho_rw
    sine_amplitude = generation_cfg.sine_amplitude
    sine_period = generation_cfg.sine_period
    background_phi = generation_cfg.background_phi
    background_rho = generation_cfg.background_rho


def generate_background_data_without_noise(settings_cfg, generation_cfg, mode='train'):
    num_testing_batches = settings_cfg.num_testing_batches
    num_data_point_per_batch = settings_cfg.num_data_point_per_batch
    num_training_data_points = settings_cfg.num_training_data_points
    if mode == 'train':
        total_testing_data_points = num_training_data_points
    elif mode == 'test':
        total_testing_data_points = num_data_point_per_batch * num_testing_batches
    else:
        total_testing_data_points = num_data_point_per_batch * num_testing_batches + num_training_data_points

    n = max(int(total_testing_data_points), 100)
    num_sensors = settings_cfg.num_sensors
    # num_training_data_points = settings_cfg.num_training_data_points

    # poisson_k = generation_cfg.poisson_k
    # poisson_lambda = generation_cfg.poisson_lambda
    # randomwalk_sd = generation_cfg.randomwalk_sd
    # background_rho_rw = generation_cfg.background_rho_rw
    # sine_amplitude = generation_cfg.sine_amplitude
    # sine_period = generation_cfg.sine_period
    # background_phi = generation_cfg.background_phi
    # background_rho = generation_cfg.background_rho

    matrix = np.zeros((total_testing_data_points, num_sensors))

    cov = np.zeros((num_sensors, num_sensors))
    for sensor_index in range(num_sensors):
        cov[sensor_index, sensor_index] = 1
    # for sensor_index in range(num_sensors):
    for key, correlated_sensor_group in settings_cfg.correlated_sensor_groups.items():
        print(key, correlated_sensor_group)
        if key == 'random_walk_groups':
            print('Random walk groups', correlated_sensor_group)
            background_type = 'random_walk'
            background_type = BACKGROUND_TYPE_MAP[background_type]
            if background_type == "Random Walk":
                # cov = np.zeros((num_sensors, num_sensors))
                # for sensor_index in range(num_sensors):
                #     cov[sensor_index, sensor_index] = 1
                randomwalk_sd = correlated_sensor_group['randomwalk_sd']
                num_random_walk_sensors = set()
                for key, val in correlated_sensor_group.groups.items():
                    sensor_ids = val['sensor_ids']
                    background_rho_rw = val['background_rho_rw']
                    # randomwalk_sd = val['randomwalk_sd']
                    i, j = sensor_ids
                    num_random_walk_sensors.add(i)
                    num_random_walk_sensors.add(j)
                    # for i, j in itertools.combinations(sensor_ids, 2):
                    cov[i, j] = background_rho_rw
                    cov[j, i] = background_rho_rw
                    # cov = np.array([[1, background_rho_rw],
                    #                 [background_rho_rw, 1]])
                num_random_walk_sensors = list(num_random_walk_sensors)
                steps = multivariate_normal(mean=np.zeros(len(num_random_walk_sensors)), cov=cov[num_random_walk_sensors, num_random_walk_sensors]).rvs(size=n)
                steps = steps * randomwalk_sd

                for index, sensor_index in enumerate(num_random_walk_sensors):
                    s1 = np.cumsum(steps[:, index])
                    # s2 = np.cumsum(steps[:, 1])
                    matrix[:, sensor_index] = s1
                # return dict(sensor1=s1, sensor2=s2)
        elif key == 'ar_1_process_groups':
            print('AR 1 process groups', correlated_sensor_group)
            background_type = 'ar_1_process'
            background_type = BACKGROUND_TYPE_MAP[background_type]
            if background_type == "AR(1) Process":
                for group_id, group in correlated_sensor_group.groups.items():
                    s1 = np.zeros(n)
                    s2 = np.zeros(n)

                    # Background Cross-Sensor Correlation
                    background_rho = group['background_rho']

                    #Background AR(1) Autocorrelation
                    background_phi = group['background_phi']
                    sensor_ids = group.sensor_ids
                    if background_rho == 1:
                        init_val = np.random.normal(0, np.sqrt(1 / (1 - background_phi ** 2)))
                        s1[0], s2[0] = init_val, init_val
                        eps = np.random.normal(size=n)
                        for t in range(1, n):
                            s1[t] = background_phi * s1[t - 1] + eps[t]
                            s2[t] = background_phi * s2[t - 1] + eps[t]
                    else:
                        cov = np.array([[1, background_rho],
                                        [background_rho, 1]])
                        init_cov = cov / (1 - background_phi ** 2)
                        init_vals = multivariate_normal(mean=[0, 0], cov=init_cov).rvs()
                        s1[0], s2[0] = init_vals
                        innovations = multivariate_normal(mean=[0, 0], cov=cov).rvs(size=n)
                        for t in range(1, n):
                            s1[t] = background_phi * s1[t - 1] + innovations[t, 0]
                            s2[t] = background_phi * s2[t - 1] + innovations[t, 1]

                # sensor_index = 0
                    matrix[:, sensor_ids[0]] = s1
                    matrix[:, sensor_ids[1]] = s2
                # return dict(sensor1=s1, sensor2=s2)
        elif key == 'independent_groups':
            for group_id, group in correlated_sensor_group.items():
                print(key, correlated_sensor_group)
                background_type = group.background_type
                background_type = BACKGROUND_TYPE_MAP[background_type]
                if background_type == "Poisson Moving Average":
                    # Poisson Noise Level
                    poisson_lambda = group['poisson_lambda']
                    # poisson_lambda = generation_cfg.sensor_data_generation_global_config.poisson_moving_average.background.poisson_noise_level
                    # Moving Average Window (k)
                    poisson_k = group['poisson_k']
                    # poisson_k = generation_cfg.sensor_data_generation_global_config.poisson_moving_average.background.moving_average_window
                    sensor_id = group.sensor_id
                    moving = np.random.poisson(lam=poisson_lambda, size=n + poisson_k)
                    # moving average (simple)
                    ma = pd.Series(moving).rolling(window=poisson_k).mean().dropna().values
                    ma = ma[:n]

                    # for sensor_index in range(num_sensors):
                    matrix[:, sensor_id] = ma
                    # return dict(sensor1=ma, sensor2=ma)

                elif background_type == "Sine Wave":
                    sine_period_per_1000 = group['sine_period']
                    # number_of_1k = total_testing_data_points // 1000
                    # sine_period = sine_period_per_1000/number_of_1k
                    sine_amplitude = group['sine_amplitude']
                    sensor_id = group.sensor_id
                    x = np.arange(1, n + 1)
                    # sw = sine_amplitude * np.sin(2 * np.pi * x / sine_period)
                    sw = sine_amplitude * np.sin(2 * np.pi * x / sine_period_per_1000)

                    matrix[:, sensor_id] = sw
                # return dict(sensor1=sw, sensor2=sw)
    # selected_index = 0

    # else:
    #     raise ValueError("Unknown background_type")

    return matrix

def get_anomaly_noise_config_by_background_type(generation_config, background_type):
    config = generation_config.sensor_data_generation_global_config[background_type]
    return config.anomaly_noise.spike_size_range, \
            config.anomaly_noise.drift_slope_range, \

def get_min_amp_and_max_amp_for_all_sensors(generation_config, background_types):
    min_amp_list = []
    max_amp_list = []

    for background_type in background_types:
        config = generation_config.sensor_data_generation_global_config[background_type]
        noise_sd = config.normal_noise.noise_sd
        noise_mean = config.normal_noise.noise_mean
        spike_size_range = config.anomaly_noise.spike_size_range
        spike_size = np.random.choice(spike_size_range)
        min_amp_list.append(2*noise_sd)
        max_amp_list.append(max(noise_mean * spike_size, abs(noise_mean) * 1))

    return min_amp_list, max_amp_list
    # min_amp_list = [2 * sd for sd in noise_sd_list]
    # max_amp_list = [max(noise_mean_list[sensor_index] * spike_size, abs(noise_mean_list[sensor_index]) * 1) for sensor_index in
    #                 range(num_sensors)]

def generate_train_and_test_data_at_once_not_mixing_anomalies(settings_cfg, generation_cfg):
    num_sensors = settings_cfg.num_sensors
    noise_mean_list = [generation_cfg.global_noise_mean for _ in range(num_sensors)]
    noise_sd_list = [generation_cfg.global_noise_sd for _ in range(num_sensors)]
    crosscor_noise = generation_cfg.global_crosscor_noise

    num_data_point_per_batch = settings_cfg.num_data_point_per_batch
    num_testing_batches = settings_cfg.num_testing_batches
    num_training_data_points = settings_cfg.num_training_data_points
    # training_data = generate_background_function(settings_cfg, generation_cfg, mode='train')

    max_independent_spikes = settings_cfg.max_independent_spikes*num_testing_batches
    max_correlated_spikes = settings_cfg.max_correlated_spikes*num_testing_batches
    max_independent_drifts = settings_cfg.max_independent_drifts*num_testing_batches
    max_correlated_drifts = settings_cfg.max_correlated_drifts*num_testing_batches
    # max_spike_length = settings_cfg.max_spike_length
    # drift_duration = settings_cfg.drift_duration
    # drift_slope = settings_cfg.drift_slope
    # spike_size = settings_cfg.spike_size

    columns = [f'Sensor{i}' for i in range(num_sensors)]
    # training_df = pd.DataFrame(training_data, columns=columns)
    df = pd.DataFrame(np.zeros((num_data_point_per_batch*num_testing_batches+ num_training_data_points, num_sensors)), columns=columns)
    # df['Time'] = np.arange(num_data_point_per_batch*num_testing_batches + num_training_data_points)
    df['Date'] = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(num_data_point_per_batch*num_testing_batches + num_training_data_points)]
    # df.set_index('Date', inplace=True)
    for index, c in enumerate(columns):
        df[f'AnomalyFlag{index}'] = new_flag_vec(num_data_point_per_batch*num_testing_batches + num_training_data_points)

    background_types = validate_settings_cfg(settings_cfg)
    # Background
    if generation_cfg.add_background:
        log.info(f'Generate background MTS')
        bg = generate_background_data_without_noise(settings_cfg, generation_cfg, mode='both')

        for sensor_index in range(num_sensors):
            s1_bg = np.array(bg[:, sensor_index])
            # s2_bg = np.array(bg["sensor2"])

            if sensor_index in settings_cfg.delayed_sensors:
                df[f'Sensor{sensor_index}'] = lagged_ema(s1_bg, generation_cfg.alpha_ema)
            else:
                df[f'Sensor{sensor_index}'] = s1_bg

            df[f'Measurand{sensor_index}'] = s1_bg + noise_mean_list[sensor_index]
    else:
        for sensor_index in range(num_sensors):
            df[f'Measurand{sensor_index}'] = noise_mean_list[sensor_index]

    print(df.head())

    # Add noise to background
    cov = np.zeros((num_sensors, num_sensors))
    for sensor_index in range(num_sensors):
        cov[sensor_index, sensor_index] = noise_sd_list[sensor_index] ** 2
    for i, j in itertools.combinations(range(num_sensors), 2):
        cov[i, j] = crosscor_noise * noise_sd_list[i] * noise_sd_list[j]
        cov[j, i] = crosscor_noise * noise_sd_list[i] * noise_sd_list[j]
    # for key, value in settings_cfg.correlated_sensor_groups.items():
    #     sensor_background_type = key[:-7]
    #     for group_order, group in value.items():
    #         print(group_order, group)
    #         sensor_ids = group['sensor_ids']
    #         for i, j in itertools.combinations(sensor_ids, 2):
    #             # cov[i,i] = sd_list[i]**2
    #             # cov[j,j] = sd_list[j]**2
    #             cov[i, j] = crosscor_noise * sd_list[i] * sd_list[j]
    #             cov[j, i] = crosscor_noise * sd_list[i] * sd_list[j]

    noise_mean = multivariate_normal(mean=[0]*num_sensors, cov=cov).rvs(size=num_data_point_per_batch*num_testing_batches+ num_training_data_points)
    log.info(f'Noise are generated with shape {noise_mean.shape}')

    for sensor_index in range(num_sensors):
        noise_mean[:, sensor_index] += (noise_mean_list[sensor_index] - noise_mean[:, sensor_index].mean())

    for sensor_index in range(num_sensors):
        df[f'Sensor{sensor_index}'] += noise_mean[:, sensor_index]

    training_df = df.iloc[:num_training_data_points, :]
    # df = None
    df = df.iloc[num_training_data_points:, :]
    df.reset_index(drop=True, inplace=True)

    max_length = df.shape[0]

    assert df.shape[0] == num_testing_batches* num_data_point_per_batch

    # min_amp_list = [2 * sd for sd in noise_sd_list]
    # max_amp_list = [max(noise_mean_list[sensor_index] * spike_size, abs(noise_mean_list[sensor_index]) * 1) for sensor_index in
    #                 range(num_sensors)]
    min_amp_list, max_amp_list = get_min_amp_and_max_amp_for_all_sensors(generation_cfg, background_types)
    def draw_amp(minv, maxv):
        return np.random.uniform(minv, maxv)

    drift_duration = generation_cfg.global_drift_duration
    max_spike_length = generation_cfg.global_max_spike_length

    number_of_anomaly_types = [1,2,3,4]
    anomaly_types = [f.name for f in ANOMALY_TYPE]



    for batch_id in range(num_testing_batches):
        num_anomaly_type_in_batch = np.random.choice(number_of_anomaly_types)
        selected_anomaly_types = np.random.choice(anomaly_types, size=num_anomaly_type_in_batch, replace=False)
        start = num_data_point_per_batch * batch_id
        end = num_data_point_per_batch * batch_id + num_data_point_per_batch

        for key, val in settings_cfg.correlated_anomaly_groups.items():
            sensor_ids = val['sensor_ids']
            log.info(f'Generate MTS for sensor groups {sensor_ids}')

            n_spikes_corr = np.random.randint(num_testing_batches, max_correlated_spikes)
            max_spike_length = generation_cfg.global_max_spike_length
            if n_spikes_corr > 0:
                starts = pick_spike_starts(max_length, n_spikes_corr,max_spike_length,  buffer=100)
                for st in starts:
                    ln = np.random.randint(1, max_spike_length + 1)
                    ed = min(st + ln - 1, max_length)
                    u = np.random.rand()
                    # u = np.random.rand()*0.5 + 0.5
                    for id in sensor_ids:
                        amp = min_amp_list[id] + u * (max_amp_list[id] - min_amp_list[id])
                        # amp1 = min_amp1 + u * (max_amp1 - min_amp1)
                        # amp2 = min_amp2 + u * (max_amp2 - min_amp2)
                        sgn = np.random.choice([-1, 1])
                        df.loc[st - 1:ed - 1, f'Sensor{id}'] += sgn * amp
                        # df.loc[st-1:ed-1, "Sensor2"] += sgn * amp2
                        df[f'AnomalyFlag{id}'] = update_flags(df[f'AnomalyFlag{id}'], range(st, ed + 1), "SpikeCorr")
                        # df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed+1), "SpikeCorr")

            n_drift_corr = np.random.randint(num_testing_batches, max_correlated_drifts)
            for st in range(n_drift_corr):
                duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
                # duration2 = np.random.randint(drift_duration[0], drift_duration[1] + 1)

                seg = pick_segment(max_length, duration, buffer=100)
                if seg is None:
                    continue
                st, ed = seg
                for id in sensor_ids:
                    spike_size_range, drift_slope_range = get_anomaly_noise_config_by_background_type(generation_cfg,  background_types[id])
                    slope_direction = np.random.choice([-1, 1])
                    if slope_direction < 0:
                        drift_slope = drift_slope_range[:2]
                    else:
                        drift_slope = drift_slope_range[-2:]

                    slope = np.random.choice(drift_slope)
                    drift = np.linspace(0, slope, duration)
                    if background_types[id] == 'ar_1_process':
                        drift *= noise_mean_list[id]

                    df.loc[st - 1:ed - 1, f'Sensor{id}'] += drift
                    # df.loc[st - 1:ed - 1, "Sensor2"] += drift
                    df[f'AnomalyFlag{id}'] = update_flags(df[f'AnomalyFlag{id}'], range(st, ed + 1), "DriftCorr")
                    # df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed + 1), "DriftCorr")

        for sensor_index in range(num_sensors):
            background_type = background_types[sensor_index]

            if ANOMALY_TYPE.SPIKE in BACKGROUND_TYPE_WITH_SUPPORTED_ANOMALY_TYPES[background_type]:
                n_spikes_s1 = np.random.randint(num_testing_batches, max_independent_spikes)
                if n_spikes_s1 > 0:
                    starts = pick_spike_starts(max_length, n_spikes_s1, max_spike_length, buffer=100)
                    for st in starts:
                        ln = np.random.randint(1, max_spike_length + 1)
                        ed = min(st + ln - 1, max_length)
                        amp = draw_amp(min_amp_list[sensor_index], max_amp_list[sensor_index])
                        sgn = np.random.choice([-1, 1])
                        df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += sgn * amp
                        df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                    "Spike")
            if ANOMALY_TYPE.DRIFT in BACKGROUND_TYPE_WITH_SUPPORTED_ANOMALY_TYPES[background_type]:
                n_drifts = np.random.randint(num_testing_batches, max_independent_drifts)
                for _ in range(n_drifts):
                    spike_size_range, drift_slope_range = get_anomaly_noise_config_by_background_type(generation_cfg,  background_types[sensor_index])
                    slope_direction = np.random.choice([-1, 1])
                    if slope_direction < 0:
                        drift_slope = drift_slope_range[:2]
                    else:
                        drift_slope = drift_slope_range[-2:]

                    # drift_slope = np.random.choice(drift_slope)
                    duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
                    slope = np.random.uniform(drift_slope[0], drift_slope[1])
                    seg = pick_segment(max_length, duration, buffer=100)
                    if seg is None:
                        continue
                    st, ed = seg
                    drift = np.linspace(0, slope, duration)
                    df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
                    df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                "Drift")

    df['batch_id'] = df.index//num_data_point_per_batch
    return df, training_df
def generate_train_and_test_data_at_once(settings_cfg, generation_cfg):
    num_sensors = settings_cfg.num_sensors
    noise_mean_list = [generation_cfg.global_noise_mean for _ in range(num_sensors)]
    noise_sd_list = [generation_cfg.global_noise_sd for _ in range(num_sensors)]
    crosscor_noise = generation_cfg.global_crosscor_noise

    num_data_point_per_batch = settings_cfg.num_data_point_per_batch
    num_testing_batches = settings_cfg.num_testing_batches
    num_training_data_points = settings_cfg.num_training_data_points
    # training_data = generate_background_function(settings_cfg, generation_cfg, mode='train')

    max_independent_spikes = settings_cfg.max_independent_spikes*num_testing_batches
    max_correlated_spikes = settings_cfg.max_correlated_spikes*num_testing_batches
    max_independent_drifts = settings_cfg.max_independent_drifts*num_testing_batches
    max_correlated_drifts = settings_cfg.max_correlated_drifts*num_testing_batches
    # max_spike_length = settings_cfg.max_spike_length
    # drift_duration = settings_cfg.drift_duration
    # drift_slope = settings_cfg.drift_slope
    # spike_size = settings_cfg.spike_size

    columns = [f'Sensor{i}' for i in range(num_sensors)]
    # training_df = pd.DataFrame(training_data, columns=columns)
    df = pd.DataFrame(np.zeros((num_data_point_per_batch*num_testing_batches+ num_training_data_points, num_sensors)), columns=columns)
    # df['Time'] = np.arange(num_data_point_per_batch*num_testing_batches + num_training_data_points)
    df['Date'] = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(num_data_point_per_batch*num_testing_batches + num_training_data_points)]
    # df.set_index('Date', inplace=True)
    for index, c in enumerate(columns):
        df[f'AnomalyFlag{index}'] = new_flag_vec(num_data_point_per_batch*num_testing_batches + num_training_data_points)

    background_types = validate_settings_cfg(settings_cfg)
    # Background
    if generation_cfg.add_background:
        log.info(f'Generate background MTS')
        bg = generate_background_data_without_noise(settings_cfg, generation_cfg, mode='both')

        for sensor_index in range(num_sensors):
            s1_bg = np.array(bg[:, sensor_index])
            # s2_bg = np.array(bg["sensor2"])

            if sensor_index in settings_cfg.delayed_sensors:
                df[f'Sensor{sensor_index}'] = lagged_ema(s1_bg, generation_cfg.alpha_ema)
            else:
                df[f'Sensor{sensor_index}'] = s1_bg

            df[f'Measurand{sensor_index}'] = s1_bg + noise_mean_list[sensor_index]
    else:
        for sensor_index in range(num_sensors):
            df[f'Measurand{sensor_index}'] = noise_mean_list[sensor_index]

    print(df.head())

    # Add noise to background
    cov = np.zeros((num_sensors, num_sensors))
    for sensor_index in range(num_sensors):
        cov[sensor_index, sensor_index] = noise_sd_list[sensor_index] ** 2
    for i, j in itertools.combinations(range(num_sensors), 2):
        cov[i, j] = crosscor_noise * noise_sd_list[i] * noise_sd_list[j]
        cov[j, i] = crosscor_noise * noise_sd_list[i] * noise_sd_list[j]
    # for key, value in settings_cfg.correlated_sensor_groups.items():
    #     sensor_background_type = key[:-7]
    #     for group_order, group in value.items():
    #         print(group_order, group)
    #         sensor_ids = group['sensor_ids']
    #         for i, j in itertools.combinations(sensor_ids, 2):
    #             # cov[i,i] = sd_list[i]**2
    #             # cov[j,j] = sd_list[j]**2
    #             cov[i, j] = crosscor_noise * sd_list[i] * sd_list[j]
    #             cov[j, i] = crosscor_noise * sd_list[i] * sd_list[j]

    noise_mean = multivariate_normal(mean=[0]*num_sensors, cov=cov).rvs(size=num_data_point_per_batch*num_testing_batches+ num_training_data_points)
    log.info(f'Noise are generated with shape {noise_mean.shape}')

    for sensor_index in range(num_sensors):
        noise_mean[:, sensor_index] += (noise_mean_list[sensor_index] - noise_mean[:, sensor_index].mean())

    for sensor_index in range(num_sensors):
        df[f'Sensor{sensor_index}'] += noise_mean[:, sensor_index]

    training_df = df.iloc[:num_training_data_points, :]
    # df = None
    df = df.iloc[num_training_data_points:, :]
    df.reset_index(drop=True, inplace=True)

    max_length = df.shape[0]

    assert df.shape[0] == num_testing_batches* num_data_point_per_batch

    # min_amp_list = [2 * sd for sd in noise_sd_list]
    # max_amp_list = [max(noise_mean_list[sensor_index] * spike_size, abs(noise_mean_list[sensor_index]) * 1) for sensor_index in
    #                 range(num_sensors)]
    min_amp_list, max_amp_list = get_min_amp_and_max_amp_for_all_sensors(generation_cfg, background_types)
    def draw_amp(minv, maxv):
        return np.random.uniform(minv, maxv)

    drift_duration = generation_cfg.global_drift_duration
    max_spike_length = generation_cfg.global_max_spike_length

    for key, val in settings_cfg.correlated_anomaly_groups.items():
        sensor_ids = val['sensor_ids']
        log.info(f'Generate MTS for sensor groups {sensor_ids}')

        n_spikes_corr = np.random.randint(num_testing_batches, max_correlated_spikes)
        max_spike_length = generation_cfg.global_max_spike_length
        if n_spikes_corr > 0:
            starts = pick_spike_starts(max_length, n_spikes_corr,max_spike_length,  buffer=100)
            for st in starts:
                ln = np.random.randint(1, max_spike_length + 1)
                ed = min(st + ln - 1, max_length)
                u = np.random.rand()
                # u = np.random.rand()*0.5 + 0.5
                for id in sensor_ids:
                    amp = min_amp_list[id] + u * (max_amp_list[id] - min_amp_list[id])
                    # amp1 = min_amp1 + u * (max_amp1 - min_amp1)
                    # amp2 = min_amp2 + u * (max_amp2 - min_amp2)
                    sgn = np.random.choice([-1, 1])
                    df.loc[st - 1:ed - 1, f'Sensor{id}'] += sgn * amp
                    # df.loc[st-1:ed-1, "Sensor2"] += sgn * amp2
                    df[f'AnomalyFlag{id}'] = update_flags(df[f'AnomalyFlag{id}'], range(st, ed + 1), "SpikeCorr")
                    # df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed+1), "SpikeCorr")

        n_drift_corr = np.random.randint(num_testing_batches, max_correlated_drifts)
        for st in range(n_drift_corr):
            duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
            # duration2 = np.random.randint(drift_duration[0], drift_duration[1] + 1)

            seg = pick_segment(max_length, duration, buffer=100)
            if seg is None:
                continue
            st, ed = seg
            for id in sensor_ids:
                spike_size_range, drift_slope_range = get_anomaly_noise_config_by_background_type(generation_cfg,  background_types[id])
                slope_direction = np.random.choice([-1, 1])
                if slope_direction < 0:
                    drift_slope = drift_slope_range[:2]
                else:
                    drift_slope = drift_slope_range[-2:]

                slope = np.random.choice(drift_slope)
                drift = np.linspace(0, slope, duration)
                if background_types[id] == 'ar_1_process':
                    drift *= noise_mean_list[id]

                df.loc[st - 1:ed - 1, f'Sensor{id}'] += drift
                # df.loc[st - 1:ed - 1, "Sensor2"] += drift
                df[f'AnomalyFlag{id}'] = update_flags(df[f'AnomalyFlag{id}'], range(st, ed + 1), "DriftCorr")
                # df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed + 1), "DriftCorr")

    for sensor_index in range(num_sensors):
        background_type = background_types[sensor_index]

        if ANOMALY_TYPE.SPIKE in BACKGROUND_TYPE_WITH_SUPPORTED_ANOMALY_TYPES[background_type]:
            n_spikes_s1 = np.random.randint(num_testing_batches, max_independent_spikes)
            if n_spikes_s1 > 0:
                starts = pick_spike_starts(max_length, n_spikes_s1, max_spike_length, buffer=100)
                for st in starts:
                    ln = np.random.randint(1, max_spike_length + 1)
                    ed = min(st + ln - 1, max_length)
                    amp = draw_amp(min_amp_list[sensor_index], max_amp_list[sensor_index])
                    sgn = np.random.choice([-1, 1])
                    df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += sgn * amp
                    df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                "Spike")
        if ANOMALY_TYPE.DRIFT in BACKGROUND_TYPE_WITH_SUPPORTED_ANOMALY_TYPES[background_type]:
            n_drifts = np.random.randint(num_testing_batches, max_independent_drifts)
            for _ in range(n_drifts):
                spike_size_range, drift_slope_range = get_anomaly_noise_config_by_background_type(generation_cfg,  background_types[sensor_index])
                slope_direction = np.random.choice([-1, 1])
                if slope_direction < 0:
                    drift_slope = drift_slope_range[:2]
                else:
                    drift_slope = drift_slope_range[-2:]

                # drift_slope = np.random.choice(drift_slope)
                duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
                slope = np.random.uniform(drift_slope[0], drift_slope[1])
                seg = pick_segment(max_length, duration, buffer=100)
                if seg is None:
                    continue
                st, ed = seg
                drift = np.linspace(0, slope, duration)
                df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
                df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                            "Drift")

    df['batch_id'] = df.index//num_data_point_per_batch
    return df, training_df

# def generate_single_batch(settings_cfg, generation_cfg, sd_list, mean_list):
#     crosscor_noise = generation_cfg.global_crosscor_noise
#     num_sensors = settings_cfg.num_sensors
#     num_data_point_per_batch = settings_cfg.num_data_point_per_batch
#     num_training_points = settings_cfg.num_training_points
#     training_data = generate_background_function(settings_cfg, generation_cfg)
#
#     max_independent_spikes = settings_cfg.max_independent_spikes
#     max_correlated_spikes = settings_cfg.max_correlated_spikes
#     max_independent_drifts = settings_cfg.max_independent_drifts
#     max_correlated_drifts = settings_cfg.max_correlated_drifts
#     max_spike_length = settings_cfg.max_spike_length
#     drift_duration = settings_cfg.drift_duration
#     drift_slope = settings_cfg.drift_slope
#     spike_size = settings_cfg.spike_size
#
#     columns = [f'Sensor{i}' for i in range(num_sensors)]
#     df = pd.DataFrame(np.zeros((num_data_point_per_batch, num_sensors)), columns=columns)
#     df['Time'] = np.arange(num_data_point_per_batch)
#     df['Date'] = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(num_data_point_per_batch)]
#     # df.set_index('Date', inplace=True)
#     for index, c in enumerate(columns):
#         df[f'AnomalyFlag{index}'] = new_flag_vec(num_data_point_per_batch)
#
#
#
#     # Background
#     if generation_cfg.add_background:
#         bg = generate_background_function(settings_cfg, generation_cfg)
#
#         for sensor_index in range(num_sensors):
#             s1_bg = np.array(bg[:, sensor_index])
#             # s2_bg = np.array(bg["sensor2"])
#
#             if sensor_index in settings_cfg.delayed_sensors:
#                 df[f'Sensor{sensor_index}'] = lagged_ema(s1_bg, generation_cfg.alpha_ema)
#             else:
#                 df[f'Sensor{sensor_index}'] = s1_bg
#
#             df[f'Measurand{sensor_index}'] = s1_bg + mean_list[sensor_index]
#     else:
#         for sensor_index in range(num_sensors):
#             df[f'Measurand{sensor_index}'] = mean_list[sensor_index]
#
#     # print(df.head())
#
#     cov = np.zeros((num_sensors, num_sensors))
#     for sensor_index in range(num_sensors):
#         cov[sensor_index, sensor_index] = sd_list[sensor_index] ** 2
#     for key, value in settings_cfg.correlated_groups.items():
#         sensor_ids = value['sensor_ids']
#         for i, j in itertools.combinations(sensor_ids, 2):
#             # cov[i,i] = sd_list[i]**2
#             # cov[j,j] = sd_list[j]**2
#             cov[i, j] = crosscor_noise * sd_list[i] * sd_list[j]
#             cov[j, i] = crosscor_noise * sd_list[i] * sd_list[j]
#
#     noise = multivariate_normal(mean=[0, 0, 0, 0, 0, 0], cov=cov).rvs(size=num_data_point_per_batch)
#     # print(noise)
#
#     for sensor_index in range(num_sensors):
#         noise[:, sensor_index] += (mean_list[sensor_index] - noise[:, sensor_index].mean())
#
#     for sensor_index in range(num_sensors):
#         df[f'Sensor{sensor_index}'] += noise[:, sensor_index]
#
#     min_amp_list = [2 * sd for sd in sd_list]
#     max_amp_list = [max(mean_list[sensor_index] * spike_size, abs(mean_list[sensor_index]) * 1) for sensor_index in
#                     range(num_sensors)]
#
#     def draw_amp(minv, maxv):
#         return np.random.uniform(minv, maxv)
#
#     for key, val in settings_cfg.correlated_groups.items():
#         sensor_ids = val['sensor_ids']
#         n_spikes_corr = np.random.randint(1, max_correlated_spikes)
#         if n_spikes_corr > 0:
#             starts = pick_spike_starts(num_data_point_per_batch, n_spikes_corr, max_spike_length, buffer=100)
#             for st in starts:
#                 ln = np.random.randint(1, max_spike_length + 1)
#                 ed = min(st + ln - 1, num_data_point_per_batch)
#                 u = np.random.rand()
#                 for id in sensor_ids:
#                     amp = min_amp_list[id] + u * (max_amp_list[id] - min_amp_list[id])
#                     # amp1 = min_amp1 + u * (max_amp1 - min_amp1)
#                     # amp2 = min_amp2 + u * (max_amp2 - min_amp2)
#                     sgn = np.random.choice([-1, 1])
#                     df.loc[st - 1:ed - 1, f'Sensor{id}'] += sgn * amp
#                     # df.loc[st-1:ed-1, "Sensor2"] += sgn * amp2
#                     df[f'AnomalyFlag{id}'] = update_flags(df[f'AnomalyFlag{id}'], range(st, ed + 1), "SpikeCorr")
#                     # df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed+1), "SpikeCorr")
#
#         n_drift_corr = np.random.randint(1, max_correlated_drifts)
#         for st in range(n_drift_corr):
#             duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
#             # duration2 = np.random.randint(drift_duration[0], drift_duration[1] + 1)
#             slope = np.random.uniform(drift_slope[0], drift_slope[1])
#             seg = pick_segment(num_data_point_per_batch, duration, buffer=100)
#             if seg is None:
#                 continue
#             st, ed = seg
#             drift = np.linspace(0, slope, duration)
#             for id in sensor_ids:
#                 df.loc[st - 1:ed - 1, f'Sensor{id}'] += drift
#                 # df.loc[st - 1:ed - 1, "Sensor2"] += drift
#                 df[f'AnomalyFlag{id}'] = update_flags(df[f'AnomalyFlag{id}'], range(st, ed + 1), "DriftCorr")
#                 # df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed + 1), "DriftCorr")
#
#     for sensor_index in range(num_sensors):
#         n_spikes_s1 = np.random.randint(1, max_independent_spikes)
#         if n_spikes_s1 > 0:
#             starts = pick_spike_starts(num_data_point_per_batch, n_spikes_s1, max_spike_length, buffer=100)
#             for st in starts:
#                 ln = np.random.randint(1, max_spike_length + 1)
#                 ed = min(st + ln - 1, num_data_point_per_batch)
#                 amp = draw_amp(min_amp_list[sensor_index], max_amp_list[sensor_index])
#                 sgn = np.random.choice([-1, 1])
#                 df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += sgn * amp
#                 df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
#                                                                 "Spike")
#
#         n_drifts = np.random.randint(1, max_independent_drifts)
#         for _ in range(n_drifts):
#             duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
#             slope = np.random.uniform(drift_slope[0], drift_slope[1])
#             seg = pick_segment(num_data_point_per_batch, duration, buffer=100)
#             if seg is None:
#                 continue
#             st, ed = seg
#             drift = np.linspace(0, slope, duration)
#             df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
#             df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
#                                                             "Drift")
#     return df
def generate_data_function_from_cfg(settings_cfg, generation_cfg):
    # print(cfg)
    # background_types = validate_settings_cfg(settings_cfg)
    num_testing_batches = settings_cfg.num_testing_batches
    num_sensors = settings_cfg.num_sensors
    num_data_point_per_batch = settings_cfg.num_data_point_per_batch
    max_independent_spikes = settings_cfg.max_independent_spikes*num_testing_batches
    max_correlated_spikes = settings_cfg.max_correlated_spikes*num_testing_batches
    max_independent_drifts = settings_cfg.max_independent_drifts*num_testing_batches
    max_correlated_drifts = settings_cfg.max_correlated_drifts*num_testing_batches
    # max_spike_length = settings_cfg.max_spike_length
    # drift_duration = settings_cfg.drift_duration
    # drift_slope = settings_cfg.drift_slope
    print(f"num_batches: {num_testing_batches}")
    print(f"num_data_point_per_batch: {num_data_point_per_batch}")
    print(f"max_independent_spikes: {max_independent_spikes}")
    print(f"max_correlated_spikes: {max_correlated_spikes}")
    print(f"max_independent_drifts: {max_independent_drifts}")
    print(f"max_correlated_drifts: {max_correlated_drifts}")

    noise_mean_range = generation_cfg.global_noise_mean
    # Noise strength in sensor
    noise_sd_range = generation_cfg.global_noise_sd


    # mean_list = [random.sample(noise_mean_range, 1)[0] for _ in range(num_sensors)]
    # sd_list = [random.sample(noise_sd_range, 1)[0] for _ in range(num_sensors)]
    # mean_list = [noise_mean_range for _ in range(num_sensors)]
    # sd_list = [noise_sd_range for _ in range(num_sensors)]

    dfs = []
    # for i in tqdm(range(num_batches), total=num_batches, desc='Generating data...'):
    #     df = generate_single_batch(settings_cfg, generation_cfg, sd_list, mean_list)
    #     dfs.append(df)
    if generation_cfg.mix_anomalies == True:
        testing_df, training_df = generate_train_and_test_data_at_once(settings_cfg, generation_cfg)
    else:
        testing_df, training_df = generate_train_and_test_data_at_once_not_mixing_anomalies(settings_cfg, generation_cfg)
    print(testing_df.shape, training_df.shape)
    dfs = [testing_df[testing_df['batch_id'] == batch_id].iloc[:,:-1] for batch_id in testing_df['batch_id'].unique()]
    dfs = [df.reset_index() for df in dfs]

    return dfs, training_df



if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')

    # my_app.py ++db.password = 1234
    main()