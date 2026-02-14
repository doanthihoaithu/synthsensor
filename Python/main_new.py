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

from anomaly_generator import ANOMALY_TYPE
from generation_manager import GenerationManager
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

# class ANOMALY_TYPE(Enum):
#     SPIKE = 1
#     CORRELATED_SPIKE = 2
#     DRIFT = 3
#     CORRELATED_DRIFT = 4

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
                # last_index = 0
                # for iteration in tqdm(range(cfg.generation.num_iterations), desc=f'Iteration ...',
                #                       total=cfg.generation.num_iterations):

                generation_manager = GenerationManager(settings_cfg, generation_cfg, add_background=True)

                generation_manager.generate_normal_data()
                generation_manager.generate_anomalies_for_all_testing_dfs()
                training_df = generation_manager.get_training_df()
                print('Training df with shape', training_df.shape)
                testing_dfs_dict = generation_manager.get_testing_dfs_dict()
                for batch_id, testing_batch in testing_dfs_dict.items():
                    print('Batch', batch_id, 'shape', testing_batch.shape)
                generation_manager.save_generated_data()
                if cfg.generation.plot_data == True:
                    generation_manager.plot_generated_data()



                # dfs, training_df = generate_data_function_from_cfg(settings_cfg, generation_cfg)
                # save_generated_data(dfs, training_df, settings_cfg, cfg, last_index)
                    # last_index += len(dfs)
def save_generated_data(testing_dfs, training_df, settings_cfg, global_cfg, last_index=0):
    # for iteration in tqdm(range(global_cfg.generation.num_iterations), desc=f'Iteration ...', total=global_cfg.generation.num_iterations):
    data_dir = global_cfg.generation.data_dir
    zip_data_dir = global_cfg.generation.zip_data_dir
    saved_file_dir = os.path.join(data_dir, settings_cfg.settings_name)
    saved_zip_file_dir = os.path.join(zip_data_dir, settings_cfg.settings_name)
    # if os.path.exists(saved_file_dir):
    #     shutil.rmtree(saved_file_dir)
    # if os.path.exists(saved_zip_file_dir):
    #     shutil.rmtree(saved_zip_file_dir)
    os.makedirs(saved_file_dir, exist_ok=True)
    os.makedirs(saved_zip_file_dir, exist_ok=True)
    training_saved_file_path = os.path.join(saved_file_dir, 'synthetic_training.csv')
    training_saved_zip_file_path = os.path.join(saved_zip_file_dir, 'synthetic_training.csv.zip')
    training_df.to_csv(training_saved_file_path, index=False)
    training_df.to_csv(training_saved_zip_file_path, index=False, compression='zip')
    log.info(f'Saved training data to {training_saved_file_path}')
    log.info(f'Saved zipped training data to {training_saved_zip_file_path}')

    for index, df in enumerate(testing_dfs):
        saved_file_path = os.path.join(saved_file_dir, f'synthetic_{index+last_index}.csv')
        saved_zip_file_path = os.path.join(saved_zip_file_dir, f'synthetic_{index+last_index}.csv.zip')
        df.to_csv(saved_file_path, index=False)
        log.info(f'Saved generated data to {saved_file_path}')
        df.to_csv(saved_zip_file_path, index=False, compression='zip')
        log.info(f'Saved zipped generated data to {saved_zip_file_path}')

        if global_cfg.generation.plot_data == True:
            fig = plot_generated_data(df, settings_cfg, global_cfg)
            saved_file_path = os.path.join(saved_file_dir, f'figure_synthetic_{index+last_index}.png')
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
            axes[i].axvline(index, color=color, alpha=0.2)

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



if __name__ == "__main__":
    print(f'Arguments: {sys.argv}')

    # my_app.py ++db.password = 1234
    main()