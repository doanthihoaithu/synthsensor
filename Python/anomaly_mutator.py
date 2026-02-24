import json

import numpy as np
import pandas as pd

from entities.sensors.data_model import ValueRange, Sensor
from synthetic_generation import pick_segment, update_flags, pick_spike_starts_from
from anomaly_generator import AnomalyGenerator, AnomalyGenerationAction, AnomalyGenerationForASingleBatch, \
    GENERATION_STRATEGY


class AnomalyMutator:
    anomaly_generator: AnomalyGenerator
    testing_dfs_dict: {str: pd.DataFrame} = dict()
    def __init__(self, anomaly_generator: AnomalyGenerator, mutation_dict: dict):
        self.anomaly_generator = anomaly_generator
        self.mutation_dict = mutation_dict
        self.sensors_dict = anomaly_generator.sensors_dict

    def enrich_specific_batch_with_anomalies(self,  actions_for_batches: {str: [AnomalyGenerationAction]}):
        actions_for_batches_enriched = dict()
        batch_ids = list(self.anomaly_generator.testing_dfs_dict.keys())
        for batch_id_to_enrich, actions in actions_for_batches.items():
            selected_batches = np.random.choice(batch_ids, size=5)

            for i, executed_batch_id in enumerate(selected_batches):
                # batch_id = list(self.anomaly_generator.testing_dfs_dict.keys())[i]
                # batch_df = self.anomaly_generator.testing_dfs_dict[batch_id].copy()
                # anomaly_generation_for_a_single_batch = AnomalyGenerationForASingleBatch(batch_df=batch_df, actions=actions)
                # anomaly_generation_for_a_single_batch.add_actions(actions)
                enriched_batch_id = f'mutation_from_{batch_id_to_enrich}_iter_{i}'
                actions_for_batches_enriched[enriched_batch_id] = (executed_batch_id, actions)
        self.generate_anomalies_for_all_batches(actions_for_batches_enriched)
        return actions_for_batches_enriched

    def generate_anomalies_for_all_batches(self, actions_for_batches: {str: [AnomalyGenerationAction]}):
        for batch_id, (batch_id_to_enrich, actions) in actions_for_batches.items():
            batch_df = self.anomaly_generator.testing_dfs_dict[batch_id_to_enrich].copy()
            anomaly_generation_for_a_single_batch = AnomalyGenerationForASingleBatch(batch_df=batch_df, actions=actions)
            # anomaly_generation_for_a_single_batch.add_actions(actions)
            self.generate_anomalies_on_batch(batch_id, anomaly_generation_for_a_single_batch)

    def generate_anomalies_on_batch(self, batch_id: str, anomaly_generation_for_a_single_batch):

        new_df = self.execute_actions_on_df(anomaly_generation_for_a_single_batch)
        self.testing_dfs_dict[batch_id] = new_df
        print(
            f'Generated anomalies on batch {batch_id} with actions: {json.dumps(anomaly_generation_for_a_single_batch, default=lambda o: o.__json__() if hasattr(o, "__json__") else None, indent=4)}')

    def execute_actions_on_df(self, single_batch_actions: AnomalyGenerationForASingleBatch) -> pd.DataFrame:
        # max_independent_spikes = 3
        # max_spike_length = 10
        df = single_batch_actions.batch_df
        actions = single_batch_actions.actions
        start = 0
        end = df.shape[0]

        # min_amp_list, max_amp_list = self.min_amp_list, self.max_amp_list

        def draw_amp(minv, maxv):
            return np.random.uniform(minv, maxv)

        for action in actions:
            if action.strategy == GENERATION_STRATEGY.ONLY_INDEPENDENT_SPIKE_ANOMALIES:
                for sensor_index in action.impacted_dimension_ids:
                    # self.inject_independent_spike_anomalies(df, sensor_index, start, end, max_independent_spikes, max_spike_length)
                    sensor: Sensor = self.anomaly_generator.get_sensor_by_id(sensor_index)
                    max_spike_length = int(sensor.spike_independent_anomaly_config.duration_range.max)
                    min_amp = sensor.get_min_amp_for_spike()
                    max_amp = sensor.get_max_amp_for_spike()

                    n_spikes_s1 = action.number_of_anomalies
                    if n_spikes_s1 > 0:
                        starts = pick_spike_starts_from(start, end, n_spikes_s1, max_spike_length, buffer=100)
                        for st in starts:
                            ln = np.random.randint(1, max_spike_length + 1)
                            ed = min(st + ln - 1, end)
                            amp = draw_amp(min_amp, max_amp)
                            sgn = np.random.choice([-1, 1])
                            df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += sgn * amp
                            df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'],
                                                                            range(st, ed + 1),
                                                                            "Spike")
            elif action.strategy == GENERATION_STRATEGY.ONLY_INDEPENDENT_DRIFT_ANOMALIES:
                n_drifts = action.number_of_anomalies
                for sensor_index in action.impacted_dimension_ids:
                    sensor: Sensor = self.anomaly_generator.get_sensor_by_id(sensor_index)
                    for _ in range(n_drifts):
                        drift_slope_range = sensor.get_slope_range_for_drift()
                        drift_duration = sensor.get_duration_range_for_drift()
                        slope_direction = np.random.choice([-1, 1])
                        if slope_direction < 0:
                            drift_slope = drift_slope_range[:2]
                        else:
                            drift_slope = drift_slope_range[-2:]

                        # drift_slope = np.random.choice(drift_slope)
                        duration = np.random.randint(drift_duration.min, drift_duration.max + 1)
                        slope = np.random.uniform(drift_slope[0], drift_slope[1])
                        seg = pick_segment(end, duration, buffer=100)
                        if seg is None:
                            continue
                        st, ed = seg
                        drift = np.linspace(slope / 2, slope, duration)
                        df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
                        df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'],
                                                                        range(st, ed + 1),
                                                                        "Drift")
            elif action.strategy == GENERATION_STRATEGY.ONLY_CORRELATED_SPIKE_ANOMALIES:
                impacted_dimension_ids = action.impacted_dimension_ids
                impacted_sensors = [self.anomaly_generator.get_sensor_by_id(sensor_index) for sensor_index in impacted_dimension_ids]
                num_anomalies = action.number_of_anomalies
                max_spike_length = int(
                    min([s.spike_independent_anomaly_config.duration_range.max for s in impacted_sensors]))
                starts = pick_spike_starts_from(start, end, num_anomalies, max_spike_length, buffer=100)
                for st in starts:
                    ln = np.random.randint(1, max_spike_length + 1)
                    ed = min(st + ln - 1, end)
                    u = np.random.rand()
                    for sensor in impacted_sensors:
                        amp = sensor.get_min_amp_for_spike() + u * (
                                    sensor.get_max_amp_for_spike() - sensor.get_min_amp_for_spike())
                        # amp2 = min_amp2 + u * (max_amp2 - min_amp2)
                        sgn = np.random.choice([-1, 1])
                        df.loc[st - 1:ed - 1, f'Sensor{sensor.id}'] += sgn * amp
                        df[f'AnomalyFlag{sensor.id}'] = update_flags(df[f'AnomalyFlag{sensor.id}'],
                                                                     range(st, ed + 1),
                                                                     "SpikeCorr")

            elif action.strategy == GENERATION_STRATEGY.ONLY_CORRELATED_DRIFT_ANOMALIES:
                impacted_dimension_ids = action.impacted_dimension_ids
                impacted_sensors = [self.anomaly_generator.get_sensor_by_id(sensor_index) for sensor_index in impacted_dimension_ids]
                num_anomalies = action.number_of_anomalies
                drift_slope_range = impacted_sensors[0].get_slope_range_for_drift()
                drift_duration = ValueRange(
                    min=max([s.drift_independent_anomaly_config.duration_range.min for s in impacted_sensors]),
                    max=min([s.drift_independent_anomaly_config.duration_range.max for s in impacted_sensors])
                )

                for _ in range(num_anomalies):
                    slope_direction = np.random.choice([-1, 1])
                    if slope_direction < 0:
                        drift_slope = drift_slope_range[:2]
                    else:
                        drift_slope = drift_slope_range[-2:]

                    # drift_slope = np.random.choice(drift_slope)
                    duration = np.random.randint(int(drift_duration.min), int(drift_duration.max) + 1)
                    slope = np.random.uniform(drift_slope[0], drift_slope[1])
                    seg = pick_segment(end, duration, buffer=100)
                    if seg is None:
                        continue
                    st, ed = seg
                    drift = np.linspace(slope / 2, slope, duration)
                    for sensor in impacted_sensors:
                        df.loc[st - 1:ed - 1, f'Sensor{sensor.id}'] += drift
                        df[f'AnomalyFlag{sensor.id}'] = update_flags(df[f'AnomalyFlag{sensor.id}'], range(st, ed + 1),
                                                                     "DriftCorr")
            elif action.strategy == GENERATION_STRATEGY.MIX_INDEPENDENT_ANOMALIES:  # min spike and drift in a dimension
                num_anomalies = action.number_of_anomalies
                num_spike_anomalies = np.random.randint(1, num_anomalies + 1)
                num_drift_anomalies = num_anomalies - num_spike_anomalies

                for sensor_index in action.impacted_dimension_ids:
                    # self.inject_independent_spike_anomalies(df, sensor_index, start, end, max_independent_spikes, max_spike_length)
                    sensor: Sensor = self.anomaly_generator.get_sensor_by_id(sensor_index)
                    max_spike_length = int(sensor.spike_independent_anomaly_config.duration_range.max)
                    min_amp = sensor.get_min_amp_for_spike()
                    max_amp = sensor.get_max_amp_for_spike()

                    n_spikes_s1 = num_spike_anomalies
                    if n_spikes_s1 > 0:
                        starts = pick_spike_starts_from(start, end, n_spikes_s1, max_spike_length, buffer=100)
                        for st in starts:
                            ln = np.random.randint(1, max_spike_length + 1)
                            ed = min(st + ln - 1, end)
                            amp = draw_amp(min_amp, max_amp)
                            sgn = np.random.choice([-1, 1])
                            df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += sgn * amp
                            df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'],
                                                                            range(st, ed + 1),
                                                                            "Spike")

                for sensor_index in action.impacted_dimension_ids:
                    sensor: Sensor = self.anomaly_generator.get_sensor_by_id(sensor_index)
                    for _ in range(num_drift_anomalies):
                        drift_slope_range = sensor.get_slope_range_for_drift()
                        drift_duration = sensor.get_duration_range_for_drift()
                        slope_direction = np.random.choice([-1, 1])
                        if slope_direction < 0:
                            drift_slope = drift_slope_range[:2]
                        else:
                            drift_slope = drift_slope_range[-2:]

                        # drift_slope = np.random.choice(drift_slope)
                        duration = np.random.randint(drift_duration.min, drift_duration.max + 1)
                        slope = np.random.uniform(drift_slope[0], drift_slope[1])
                        seg = pick_segment(end, duration, buffer=100)
                        if seg is None:
                            continue
                        st, ed = seg
                        drift = np.linspace(slope / 2, slope, duration)
                        df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
                        df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'],
                                                                        range(st, ed + 1),
                                                                        "Drift")

            elif action.strategy == GENERATION_STRATEGY.MIX_CORRELATED_ANOMALIES:  # min Correlated spike and Correlated drift in a dimension
                impacted_dimension_ids = action.impacted_dimension_ids
                impacted_sensors = [self.anomaly_generator.get_sensor_by_id(sensor_index) for sensor_index in impacted_dimension_ids]
                num_anomalies = action.number_of_anomalies
                if num_anomalies == 1:
                    continue
                num_correlated_spike_anomalies = np.random.randint(1, num_anomalies)
                num_correlated_drift_anomalies = num_anomalies - num_correlated_spike_anomalies

                max_spike_length = int(
                    min([s.spike_independent_anomaly_config.duration_range.max for s in impacted_sensors]))
                starts = pick_spike_starts_from(start, end, num_correlated_spike_anomalies, max_spike_length,
                                                buffer=100)
                for st in starts:
                    ln = np.random.randint(1, max_spike_length + 1)
                    ed = min(st + ln - 1, end)
                    u = np.random.rand()
                    for sensor in impacted_sensors:
                        amp = sensor.get_min_amp_for_spike() + u * (
                                sensor.get_max_amp_for_spike() - sensor.get_min_amp_for_spike())
                        # amp2 = min_amp2 + u * (max_amp2 - min_amp2)
                        sgn = np.random.choice([-1, 1])
                        df.loc[st - 1:ed - 1, f'Sensor{sensor.id}'] += sgn * amp
                        df[f'AnomalyFlag{sensor.id}'] = update_flags(df[f'AnomalyFlag{sensor.id}'],
                                                                     range(st, ed + 1),
                                                                     "SpikeCorr")

                drift_slope_range = impacted_sensors[0].get_slope_range_for_drift()
                drift_duration = ValueRange(
                    min=max([s.drift_independent_anomaly_config.duration_range.min for s in impacted_sensors]),
                    max=min([s.drift_independent_anomaly_config.duration_range.max for s in impacted_sensors])
                )

                for _ in range(num_correlated_drift_anomalies):
                    slope_direction = np.random.choice([-1, 1])
                    if slope_direction < 0:
                        drift_slope = drift_slope_range[:2]
                    else:
                        drift_slope = drift_slope_range[-2:]

                    # drift_slope = np.random.choice(drift_slope)
                    duration = np.random.randint(int(drift_duration.min), int(drift_duration.max) + 1)
                    slope = np.random.uniform(drift_slope[0], drift_slope[1])
                    seg = pick_segment(end, duration, buffer=100)
                    if seg is None:
                        continue
                    st, ed = seg
                    drift = np.linspace(slope / 2, slope, duration)
                    for sensor in impacted_sensors:
                        df.loc[st - 1:ed - 1, f'Sensor{sensor.id}'] += drift
                        df[f'AnomalyFlag{sensor.id}'] = update_flags(df[f'AnomalyFlag{sensor.id}'], range(st, ed + 1),
                                                                     "DriftCorr")
            elif action.strategy == GENERATION_STRATEGY.MIX_INDEPENDENT_AND_CORRELATED_ANOMALIES:
                pass
            else:
                raise NotImplementedError(f'Generation strategy {action.strategy} not implemented yet')
        return df