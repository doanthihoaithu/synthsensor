import itertools
import json
import os.path
import random
from enum import Enum
from typing import Dict, Tuple, Union, Optional

import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal


import numpy as np
import logging

from anomaly_generator import AnomalyGenerationAction, GENERATION_STRATEGY, AnomalyGenerator, \
    AnomalyGenerationActionMix
from entities.sensors.data_model import Sensor, Noise, AR_1_Process_Signal, ValueRange, ValueList, \
    Sine_Wave_Signal, Poisson_Moving_Average_Signal, CorrelationConstraint, BACKGROUND_TYPE, CORRELATION_CONSTRAINT_TYPE

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)





class GenerationManager:
    config_data: DictConfig
    sensors_dict: dict[int, Sensor]
    correlated_constraints: [CorrelationConstraint] = []
    add_background: bool = True

    def __init__(self, settings_cfg: DictConfig, generation_cfg:DictConfig, add_background: bool = True):
        self.setting_cfg = settings_cfg
        self.generation_cfg = generation_cfg
        # self.background_types = validate_settings_cfg(settings_cfg)x
        self.setting_name = settings_cfg.settings_name
        # self.num_sensors = settings_cfg.num_sensors
        self.import_sensors_from_settings_config(settings_cfg)
        self.add_background = add_background


        self.num_training_data_points: int = settings_cfg.num_training_data_points
        self.num_testing_batches: int = settings_cfg.num_testing_batches
        tmp_range = settings_cfg.num_data_point_per_batch_range
        self.num_data_point_per_testing_batch_range: tuple[int, int] = tmp_range.min, tmp_range.max
        assert self.num_data_point_per_testing_batch_range[1] > self.num_data_point_per_testing_batch_range[0]

    def import_sensors_from_settings_config(self, settings_cfg):
        ar_1_process_sensors_config = settings_cfg.ar_1_process_sensors
        ar_1_process_sensors, ar_1_correlated_groups = self.extract_ar_1_process_sensors_from_settings_cfg(ar_1_process_sensors_config)

        sine_wave_sensors_config = settings_cfg.sine_wave_sensors
        sine_wave_sensors, sine_wave_correlated_groups = self.extract_sine_wave_sensors_from_settings_cfg(sine_wave_sensors_config)

        poisson_moving_average_sensors_config = settings_cfg.poisson_moving_average_sensors
        poisson_moving_average_sensors, poisson_correlated_groups = self.extract_poisson_moving_average_sensors_from_settings_cfg(poisson_moving_average_sensors_config)

        sensors = ar_1_process_sensors + sine_wave_sensors + poisson_moving_average_sensors
        correlated_constraints = ar_1_correlated_groups + sine_wave_correlated_groups + poisson_correlated_groups
        self.correlated_constraints = correlated_constraints
        for c in correlated_constraints:
            print('correlated constraint', c)

        self.sensors_dict: dict[int, Sensor] = {sensor.id:sensor for sensor in sensors}
        self.num_sensors = len(self.sensors_dict)
        # self.background_types = [s.background_signal.background_type for s in self.sensors]
        # self.noise_mean_list = [s.normal_noise_mean for s in self.sensors]
        # self.noise_sd_list = [s.normal_noise_sd for s in self.sensors]
        # self.crosscor_noise = self.generation_cfg.global_crosscor_noise

    def get_sensor_by_id(self, sensor_id) -> Optional[Sensor]:
        return self.sensors_dict[sensor_id]
    def initialize_general_data(self):
        # self.initialize_normal_noise_data()
        # assert self.noise_mean_list is not None
        # assert self.noise_sd_list is not None
        self.init_various_number_of_data_points_for_testing_batches()
        assert len(self.testing_batch_size_list) == self.num_testing_batches

    def generate_normal_data(self):
        self.initialize_general_data()
        self.generate_background_data()

        self.add_noise_matrix_to_normal_background_data()
        #
        # self.add_anomalies_to_noise_background_data()
        self.set_training_df()
        self.set_testing_dfs_dict()

    def generate_anomalies_for_all_testing_dfs(self):
        actions_for_batches = dict()
        strategies = [
                      GENERATION_STRATEGY.ONLY_INDEPENDENT_SPIKE_ANOMALIES,
                      GENERATION_STRATEGY.ONLY_INDEPENDENT_DRIFT_ANOMALIES,
                      GENERATION_STRATEGY.MIX_INDEPENDENT_ANOMALIES,
                      ]
        impacted_dimension_ids_list = list(itertools.chain(*[itertools.combinations(range(self.num_sensors), ni) for ni in range((self.num_sensors + 1)//2)]))
        number_of_anomalies_list = [1, 2, 3]
        configurations_independent = list(itertools.product(strategies, impacted_dimension_ids_list, number_of_anomalies_list))

        correlated_strategies = [
                                    GENERATION_STRATEGY.ONLY_CORRELATED_SPIKE_ANOMALIES,
                                    GENERATION_STRATEGY.ONLY_CORRELATED_DRIFT_ANOMALIES,
                                    GENERATION_STRATEGY.MIX_CORRELATED_ANOMALIES
                                ]
        impacted_dimension_correlated_ids_list = [correlated_constraint.sensor_ids for correlated_constraint in
                                                  self.correlated_constraints
                                                  if
                                                  correlated_constraint.correlation_type == CORRELATION_CONSTRAINT_TYPE.NOISE_CROSS_CORRELATION]
        configurations_correlated = list(itertools.product(correlated_strategies, impacted_dimension_correlated_ids_list, number_of_anomalies_list[1:]))
        # configurations = configurations_correlated + configurations_independent
        # random.shuffle(configurations)
        # configurations = configurations_independent
        # configurations = configurations_independent

        generation_logs = dict()

        mix_anomalies_strategies = ['independent', 'correlated', 'mix']
        selected_strategy_for_batches = np.random.randint(0, 3, size=len(list(self.testing_dfs_dict.keys())))

        for (batch_id, batch_df), selected_strategy in zip(self.testing_dfs_dict.items(), selected_strategy_for_batches):
            # actions = [AnomalyGenerationAction(strategy=GENERATION_STRATEGY.ONLY_INDEPENDENT_SPIKE_ANOMALIES,
            #                                    impacted_dimension_ids=[0],
            #                                    number_of_anomalies=5
            #                                    )]
            selected_configurations = []
            for k in range(1,4):
                if mix_anomalies_strategies[selected_strategy] == 'independent' or mix_anomalies_strategies[selected_strategy]=='mix':
                    selected_configurations.extend(random.choices(configurations_independent, k=k))
                if mix_anomalies_strategies[selected_strategy] == 'correlated' or mix_anomalies_strategies[selected_strategy]=='mix':
                    selected_configurations.extend(random.choices(configurations_correlated, k=k))
            actions = [AnomalyGenerationAction(strategy=strategy,
                                               impacted_dimension_ids=impacted_dimension_ids,
                                               number_of_anomalies=number_of_anomalies
                                               )
                       for strategy, impacted_dimension_ids, number_of_anomalies in selected_configurations]
            # actions.extend([AnomalyGenerationActionMix(strategy=strategy,
            #                                    impacted_dimension_ids=impacted_dimension_ids,
            #                                    number_of_anomalies=number_of_anomalies,
            #                                    num_independent_spike_anomalies=1,
            #                                    num_independent_drift_anomalies=number_of_anomalies-1
            #                                    )
            #            for strategy, impacted_dimension_ids, number_of_anomalies in selected_configurations])
            actions_for_batches[batch_id] = actions
            # anomaly_generation_for_a_single_batch = AnomalyGenerationForASingleBatch(batch_df=batch_df)
            # anomaly_generation_for_a_single_batch.add_actions(actions)

        # background_types = validate_settings_cfg(self.setting_cfg)
        # min_amp_list, max_amp_list = self.get_min_amp_and_max_amp_for_all_sensors()
        anomaly_generator = AnomalyGenerator(testing_dfs_dict=self.testing_dfs_dict,
                                             sensors_dict=self.sensors_dict,
                                             )
        anomaly_generator.generate_anomalies_for_all_batches(actions_for_batches=actions_for_batches)
        save_generation_history_dir = self.generation_cfg.save_generation_history_dir
        os.makedirs(save_generation_history_dir, exist_ok=True)
        generation_history_df = pd.DataFrame(columns=['batch_id', 'strategy', 'impacted_dimension_ids', 'number_of_anomalies'])
        for batch_id, actions in actions_for_batches.items():
            new_df = pd.DataFrame(action.__dict__ for action in actions)
            new_df['batch_id'] = batch_id
            generation_history_df = pd.concat([generation_history_df, new_df], ignore_index=True)
            # for action in actions:
            #     generation_history_df.loc[len(generation_history_df)] = {
            #         'batch_id': batch_id,
            #         'strategy': action.strategy.name,
            #         'impacted_dimension_ids': action.impacted_dimension_ids,
            #         'number_of_anomalies': action.number_of_anomalies,
            #     }
        generation_history_df.to_csv(os.path.join(save_generation_history_dir, 'generation_history.csv'), index=False)

        # with open(os.path.join(save_generation_history_dir, 'generation_history.json'), 'w') as f:
        #     f.write(json.dumps(actions_for_batches,
        #                        default=lambda o: o.__json__() if hasattr(o, "__json__") else None,
        #                        indent=4))


    def init_various_number_of_data_points_for_testing_batches(self):
        batch_sizes = np.random.random_integers(low=self.num_data_point_per_testing_batch_range[0],
                                     high=self.num_data_point_per_testing_batch_range[1],
                                     size=self.num_testing_batches
                                     )
        self.testing_batch_size_list = batch_sizes


    # def initialize_normal_noise_data(self):
        # self.noise_mean_list = [s.normal_noise_mean for s in self.sensors_dict]
        # self.noise_sd_list = [s.normal_noise_sd for s in self.sensors_dict]
        # self.crosscor_noise = self.generation_cfg.global_crosscor_noise

    def add_noise_matrix_to_normal_background_data(self):
        assert self.background_normal_data is not None
        self.background_noise_data = self.background_normal_data.copy()
        num_sensors = self.num_sensors
        # Add noise to background
        cov = np.zeros((num_sensors, num_sensors))
        for sensor_index in range(num_sensors):
            cov[sensor_index, sensor_index] = self.get_sensor_by_id(sensor_index).noise.sd ** 2

        for correlated_constraint in self.correlated_constraints:
            if correlated_constraint.correlation_type == CORRELATION_CONSTRAINT_TYPE.NOISE_CROSS_CORRELATION:
                sensor_ids = correlated_constraint.sensor_ids
                sensor_id_index_mapping = correlated_constraint.sensor_id_index_mapping
                cross_correlation_matrix = correlated_constraint.cross_correlation_matrix
                for sensor_id_i, sensor_id_j in itertools.combinations(sensor_ids, 2):
                    noise_sd_i = self.get_sensor_by_id(sensor_id_i).noise.sd
                    noise_sd_j = self.get_sensor_by_id(sensor_id_j).noise.sd
                    index_i = sensor_id_index_mapping[sensor_id_i]
                    index_j = sensor_id_index_mapping[sensor_id_j]
                    cov[sensor_id_i, sensor_id_j] = cross_correlation_matrix[index_i, index_j] * noise_sd_i * noise_sd_j
                    cov[sensor_id_j, sensor_id_i] = cross_correlation_matrix[index_i, index_j] * noise_sd_i * noise_sd_j
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
        # total_data_points = self.num_training_data_points + sum(self.testing_batch_size_list)
        noise_mean = multivariate_normal(mean=[0] * num_sensors, cov=cov).rvs(
            size=self.total_data_points)
        log.info(f'Noise are generated with shape {noise_mean.shape}')

        for sensor_index in range(num_sensors):
            noise_mean[:, sensor_index] += (self.get_sensor_by_id(sensor_index).noise.mean - noise_mean[:, sensor_index].mean())

        for sensor_index in range(num_sensors):
            self.background_noise_data[f'Sensor{sensor_index}'] += noise_mean[:, sensor_index]

    def add_anomalies_to_noise_background_data(self):

        pass


    def generate_background_data(self):
        columns = [f'Sensor{i}' for i in range(self.num_sensors)]
        flag_columns = [f'AnomalyFlag{i}' for i in range(self.num_sensors)]
        self.background_normal_data = pd.DataFrame(columns=columns)
        self.total_data_points = self.num_training_data_points + sum(self.testing_batch_size_list)

        self.background_normal_data[columns] = self.generate_background_data_without_noise(mode='both')
        self.background_normal_data[flag_columns] = 'Normal'
        # self.background_noise_data= self.background_normal_data.copy()


        assert self.background_normal_data.shape[0] == self.num_training_data_points + sum(self.testing_batch_size_list)

    def set_training_df(self):
        self.training_df = self.background_noise_data.iloc[:self.num_training_data_points, :]

    def get_training_df(self):
        return self.training_df
    def set_testing_dfs_dict(self):
        dfs_dict = dict()
        start_index = self.num_training_data_points
        for i, size in enumerate(self.testing_batch_size_list):
            batch_df = self.background_noise_data.iloc[start_index:start_index + size, :]
            start_index += size
            dfs_dict[f'batch_{i}'] = batch_df.copy().reset_index()
        self.testing_dfs_dict = dfs_dict

    def get_testing_dfs_dict(self) -> Dict[str, pd.DataFrame]:
        return self.testing_dfs_dict


    def save_total_data_points_as_an_image(self):
        print('Training shape', self.training_df.shape)
        print('Testing batches info:')
        for batch_id, df in self.testing_dfs_dict.items():
            print(f'Batch id {batch_id} with shape', df.shape)

    def generate_background_data_without_noise(self, mode='train'):
        settings_cfg = self.setting_cfg
        # generation_cfg = self.generation_cfg
        # num_testing_batches = settings_cfg.num_testing_batches
        # num_data_point_per_batch = settings_cfg.num_data_point_per_batch
        # num_training_data_points = settings_cfg.num_training_data_points
        if mode == 'train':
            total_testing_data_points = self.num_training_data_points
        elif mode == 'test':
            total_testing_data_points = sum(self.testing_batch_size_list)
        else:
            total_testing_data_points = self.total_data_points

        n = max(int(total_testing_data_points), 100)
        num_sensors = self.num_sensors
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
        if self.add_background == False:
            return matrix
        is_generated_check = np.zeros(num_sensors)

        cov = np.zeros((num_sensors, num_sensors))
        for sensor_index in range(num_sensors):
            cov[sensor_index, sensor_index] = 1
        # for sensor_index in range(num_sensors):
        for ar_1_sensor_group in self.get_background_ar_1_sensor_groups():
            # if key == 'ar_1_process_groups':
            print('AR 1 process groups', ar_1_sensor_group)
            background_type = 'ar_1_process'
            # background_type = BACKGROUND_TYPE_MAP[background_type]
            # if background_type == "ar_1_process":
            # for group_id, group in correlated_sensor_group.groups.items():
            sensor_ids= ar_1_sensor_group.sensor_ids
            sensor_id_index_mapping  = ar_1_sensor_group.sensor_id_index_mapping
            cross_correlation_matrix = ar_1_sensor_group.cross_correlation_matrix
            init_matrix = np.zeros((len(sensor_ids), n))
            background_phi_matrix = np.zeros((len(sensor_ids), len(sensor_ids)))
            for sensor_id in sensor_ids:
                background_phi_matrix[sensor_id_index_mapping[sensor_id], sensor_id_index_mapping[sensor_id]] = self.get_sensor_by_id(sensor_id).background_signal.background_phi

            cov = cross_correlation_matrix
            for sensor_id in sensor_id_index_mapping.values():
                cov[sensor_id, sensor_id] = 1
            init_cov = cov / (1 - background_phi_matrix ** 2)
            init_vals = multivariate_normal(mean=[0]*len(sensor_ids), cov=init_cov).rvs()
            init_matrix[:,0] = init_vals
            innovations = multivariate_normal(mean=[0]*len(sensor_ids), cov=cov).rvs(size=n)
            for t in range(1, n):
                # for i in range(len(sensor_ids)):
                init_matrix[:, t] = background_phi_matrix.diagonal() * init_matrix[:,t-1] + innovations[t, :]

            # sensor_index = 0
            for sensor_id in sensor_ids:
                is_generated_check[sensor_id] = 1
                matrix[:, sensor_id] = init_matrix[sensor_id_index_mapping[sensor_id], :]
        for sensor in self.sensors_dict.values():
            if is_generated_check[sensor.id] == 0:
                if sensor.background_type == BACKGROUND_TYPE.SINE_WAVE:
                    background_signal: Sine_Wave_Signal = sensor.background_signal
                    sine_period_per_1000 = background_signal.background_period
                    sine_amplitude = background_signal.background_amplitude
                    sensor_id = sensor.id
                    x = np.arange(1, n + 1)
                    # sw = sine_amplitude * np.sin(2 * np.pi * x / sine_period)
                    sw = sine_amplitude * np.sin(2 * np.pi * x / sine_period_per_1000)

                    matrix[:, sensor_id] = sw
                elif sensor.background_type == BACKGROUND_TYPE.POISSON_MOVING_AVERAGE:
                    background_signal: Poisson_Moving_Average_Signal = sensor.background_signal
                    poisson_lambda = background_signal.background_lambda
                    # Moving Average Window (k)
                    poisson_k = background_signal.background_k
                    sensor_id = sensor.id
                    moving = np.random.poisson(lam=poisson_lambda, size=n + poisson_k)
                    # moving average (simple)
                    ma = pd.Series(moving).rolling(window=poisson_k).mean().dropna().values
                    ma = ma[:n]
                    # for sensor_index in range(num_sensors):
                    matrix[:, sensor_id] = ma
        return matrix

    def get_background_ar_1_sensor_groups(self)->[CorrelationConstraint]:
        return [c for c in self.correlated_constraints if c.correlation_type == CORRELATION_CONSTRAINT_TYPE.BACKGROUND_AR_1_CROSS_CORRELATION]

    def save_generated_data(self):
        generation_config = self.generation_cfg
        settings_cfg = self.setting_cfg
        data_dir = generation_config.data_dir
        zip_data_dir = generation_config.zip_data_dir
        saved_file_dir = os.path.join(data_dir)
        saved_zip_file_dir = os.path.join(zip_data_dir)
        # if os.path.exists(saved_file_dir):
        #     shutil.rmtree(saved_file_dir)
        # if os.path.exists(saved_zip_file_dir):
        #     shutil.rmtree(saved_zip_file_dir)
        os.makedirs(saved_file_dir, exist_ok=True)
        os.makedirs(saved_zip_file_dir, exist_ok=True)
        training_saved_file_path = os.path.join(saved_file_dir, 'synthetic_training.csv')
        training_saved_zip_file_path = os.path.join(saved_zip_file_dir, 'synthetic_training.csv.zip')
        self.training_df.to_csv(training_saved_file_path, index=False)
        self.training_df.to_csv(training_saved_zip_file_path, index=False, compression='zip')
        log.info(f'Saved training data to {training_saved_file_path}')
        log.info(f'Saved zipped training data to {training_saved_zip_file_path}')

        for index, (key, df) in enumerate(self.testing_dfs_dict.items()):
            saved_file_path = os.path.join(saved_file_dir, f'synthetic_{index}.csv')
            saved_zip_file_path = os.path.join(saved_zip_file_dir, f'synthetic_{index}.csv.zip')
            df.to_csv(saved_file_path, index=False)
            log.info(f'Saved generated data to {saved_file_path}')
            df.to_csv(saved_zip_file_path, index=False, compression='zip')
            log.info(f'Saved zipped generated data to {saved_zip_file_path}')

    def plot_generated_data(self):
        assert self.training_df is not None
        self._plot_train_df()

        assert len(list(self.testing_dfs_dict.keys())) > 0
        self._plot_testing_dfs_dict()

    def _plot_train_df(self):
        df = self.training_df
        num_sensors = self.num_sensors
        data_dir = self.generation_cfg.data_dir
        fig = self._get_figure_of_single_df(df, num_sensors)
        figure_path = os.path.join(data_dir, 'figures', 'synthetic_training.png')
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        fig.savefig(figure_path)
        print(f'Plot training df at ', figure_path )
        return None

    def _plot_testing_dfs_dict(self):
        num_sensors = self.num_sensors
        data_dir = self.generation_cfg.data_dir
        for key, df in self.testing_dfs_dict.items():
            fig = self._get_figure_of_single_df(df, num_sensors)
            figure_path = os.path.join(data_dir, 'figures', f'{key}.png')
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            fig.savefig(figure_path)
            print(f'Plot testing df batch_id  = {key} at ', figure_path)

    def _get_figure_of_single_df(self, df, num_sensors):
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


    def extract_ar_1_process_sensors_from_settings_cfg(self, ar_1_process_sensors_config) -> Tuple[any,any]:
        sensor_list = []
        id_list = ar_1_process_sensors_config.id_list
        background_phi_list = ar_1_process_sensors_config.background_phi_list
        noise_mean_list = ar_1_process_sensors_config.noise_mean_list
        noise_sd_list = ar_1_process_sensors_config.noise_sd_list
        spike_length_range_list = ar_1_process_sensors_config.spike_length_range_list
        spike_magnitude_range_list = ar_1_process_sensors_config.spike_magnitude_range_list
        drift_duration_range_list = ar_1_process_sensors_config.drift_duration_range_list
        drift_slope_range_list = ar_1_process_sensors_config.drift_slope_range_list
        for sensor_id, \
                background_phi, \
                noise_mean, \
                noise_sd, \
                spike_length_range, \
                spike_magnitude_range, \
                drift_duration_range, \
                drift_slope_range in zip(id_list,
                                         background_phi_list,
                                         noise_mean_list,
                                         noise_sd_list,
                                         spike_length_range_list,
                                         spike_magnitude_range_list,
                                         drift_duration_range_list,
                                         drift_slope_range_list):
            new_noise = Noise(mean=noise_mean, sd=noise_sd)
            new_sensor = Sensor(id=sensor_id, noise=new_noise)
            background_signal = AR_1_Process_Signal(background_phi=background_phi)
            new_sensor.add_background_signal(background_signal)
            new_sensor.set_spike_independent_anomaly_config(
                magnitude_range=ValueRange(min=spike_magnitude_range[0], max=spike_magnitude_range[1]),
                duration_range=ValueRange(min=spike_length_range[0], max=spike_length_range[1])
            )
            new_sensor.set_drift_independent_anomaly_config(
                duration_range=ValueRange(min=drift_duration_range[0], max=drift_duration_range[1]),
                slope_range=ValueList(drift_slope_range)
            )
            sensor_list.append(new_sensor)
        correlated_groups_dict = ar_1_process_sensors_config.correlated_groups
        correlated_constraints = self.extract_ar_1_process_cross_correlation_constraints_from_settings_cfg(correlated_groups_dict)
        # correlated_noise_contraints = self.extract_noise_cross_correlation_constraints_from_settings_cfg(correlated_groups_dict, sensors)
        return sensor_list, correlated_constraints

    def extract_sine_wave_sensors_from_settings_cfg(self, sine_wave_sensors_config)->Tuple[any, any]:
        sensor_list = []
        id_list = sine_wave_sensors_config.id_list
        sine_period_list = sine_wave_sensors_config.sine_period_list
        sine_amplitude_list = sine_wave_sensors_config.sine_amplitude_list
        noise_mean_list = sine_wave_sensors_config.noise_mean_list
        noise_sd_list = sine_wave_sensors_config.noise_sd_list
        spike_length_range_list = sine_wave_sensors_config.spike_length_range_list
        spike_magnitude_range_list = sine_wave_sensors_config.spike_magnitude_range_list
        drift_duration_range_list = sine_wave_sensors_config.drift_duration_range_list
        drift_slope_range_list = sine_wave_sensors_config.drift_slope_range_list
        for sensor_id, \
                sine_period, \
                sine_amplitude, \
                noise_mean, \
                noise_sd, \
                spike_length_range, \
                spike_magnitude_range, \
                drift_duration_range, \
                drift_slope_range in zip(id_list,
                                         sine_period_list,
                                         sine_amplitude_list,
                                         noise_mean_list,
                                         noise_sd_list,
                                         spike_length_range_list,
                                         spike_magnitude_range_list,
                                         drift_duration_range_list,
                                         drift_slope_range_list):
            new_noise = Noise(mean=noise_mean, sd=noise_sd)
            new_sensor = Sensor(id=sensor_id, noise=new_noise)
            background_signal = Sine_Wave_Signal(background_period=sine_period, background_amplitude=sine_amplitude)
            new_sensor.add_background_signal(background_signal)
            new_sensor.set_spike_independent_anomaly_config(
                magnitude_range=ValueRange(min=spike_magnitude_range[0], max=spike_magnitude_range[1]),
                duration_range=ValueRange(min=spike_length_range[0], max=spike_length_range[1])
            )
            new_sensor.set_drift_independent_anomaly_config(
                duration_range=ValueRange(min=drift_duration_range[0], max=drift_duration_range[1]),
                slope_range=ValueList(drift_slope_range)
            )
            sensor_list.append(new_sensor)
        correlated_groups_dict = sine_wave_sensors_config.correlated_groups
        correlated_constraints = self.extract_noise_cross_correlation_constraints_from_settings_cfg(correlated_groups_dict)
        return sensor_list, correlated_constraints

    def extract_poisson_moving_average_sensors_from_settings_cfg(self, poisson_moving_average_sensors_config)->Tuple[any, any]:
        sensor_list = []
        id_list = poisson_moving_average_sensors_config.id_list
        poisson_lambda_list = poisson_moving_average_sensors_config.poisson_lambda_list
        poisson_k_list = poisson_moving_average_sensors_config.poisson_k_list
        noise_mean_list = poisson_moving_average_sensors_config.noise_mean_list
        noise_sd_list = poisson_moving_average_sensors_config.noise_sd_list
        spike_length_range_list = poisson_moving_average_sensors_config.spike_length_range_list
        spike_magnitude_range_list = poisson_moving_average_sensors_config.spike_magnitude_range_list
        drift_duration_range_list = poisson_moving_average_sensors_config.drift_duration_range_list
        drift_slope_range_list = poisson_moving_average_sensors_config.drift_slope_range_list
        for sensor_id, \
                poisson_lambda, \
                poisson_k, \
                noise_mean, \
                noise_sd, \
                spike_length_range, \
                spike_magnitude_range, \
                drift_duration_range, \
                drift_slope_range in zip(id_list,
                                         poisson_lambda_list,
                                         poisson_k_list,
                                         noise_mean_list,
                                         noise_sd_list,
                                         spike_length_range_list,
                                         spike_magnitude_range_list,
                                         drift_duration_range_list,
                                         drift_slope_range_list):
            new_noise = Noise(mean=noise_mean, sd=noise_sd)
            new_sensor = Sensor(id=sensor_id, noise=new_noise)
            background_signal = Poisson_Moving_Average_Signal(background_lambda=poisson_lambda, background_k=poisson_k)
            new_sensor.add_background_signal(background_signal)
            new_sensor.set_spike_independent_anomaly_config(
                magnitude_range=ValueRange(min=spike_magnitude_range[0], max=spike_magnitude_range[1]),
                duration_range=ValueRange(min=spike_length_range[0], max=spike_length_range[1])
            )
            new_sensor.set_drift_independent_anomaly_config(
                duration_range=ValueRange(min=drift_duration_range[0], max=drift_duration_range[1]),
                slope_range=ValueList(drift_slope_range)
            )
            sensor_list.append(new_sensor)
        correlated_groups_dict = poisson_moving_average_sensors_config.correlated_groups
        correlated_constraints = self.extract_noise_cross_correlation_constraints_from_settings_cfg(correlated_groups_dict)
        return sensor_list, correlated_constraints

    def extract_ar_1_process_cross_correlation_constraints_from_settings_cfg(self, correlated_groups_dict):
        constraints = []
        for group_id, group in correlated_groups_dict.items():
            print('AR(1) Correlated Infor', group_id, group.ids, "->",group.corresponding_index, group.background_correlated_dict)
            background_correlated_dict = group.background_correlated_dict
            noise_cross_correlated_dict = group.noise_cross_correlated_dict
            background_correlated_constraint = CorrelationConstraint(sensor_ids=group.ids,
                                                                   sensor_id_index_mapping={sensor_id: mapping_index for sensor_id, mapping_index in zip(group.ids, group.corresponding_index)},
                                                                   cross_correlation_dict=background_correlated_dict,
                                                                   correlation_type=CORRELATION_CONSTRAINT_TYPE.BACKGROUND_AR_1_CROSS_CORRELATION)
            noise_correlated_constraint = CorrelationConstraint(sensor_ids=group.ids,
                                                                   sensor_id_index_mapping={sensor_id: mapping_index for sensor_id, mapping_index in zip(group.ids, group.corresponding_index)},
                                                                   cross_correlation_dict=noise_cross_correlated_dict,
                                                                   correlation_type=CORRELATION_CONSTRAINT_TYPE.NOISE_CROSS_CORRELATION)

            constraints.append(background_correlated_constraint)
            constraints.append(noise_correlated_constraint)
        return constraints


    def extract_noise_cross_correlation_constraints_from_settings_cfg(self, correlated_groups_dict):
        constraints = []
        for group_id, group in correlated_groups_dict.items():
            print('Noise Correlated Infor', group_id, group.ids,"->",group.corresponding_index, group.noise_cross_correlated_dict)
            noise_cross_correlated_dict = group.noise_cross_correlated_dict
            noise_correlated_constraint = CorrelationConstraint(sensor_ids=group.ids,
                                                                sensor_id_index_mapping={sensor_id: index for
                                                                                         index, sensor_id in
                                                                                         enumerate(group.ids)},
                                                                cross_correlation_dict=noise_cross_correlated_dict,
                                                                correlation_type=CORRELATION_CONSTRAINT_TYPE.NOISE_CROSS_CORRELATION)

            constraints.append(noise_correlated_constraint)
        return constraints
