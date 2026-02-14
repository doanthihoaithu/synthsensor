import json
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd

from entities.sensors.data_model import Sensor, ValueRange
from synthetic_generation import pick_spike_starts_from, update_flags, pick_segment


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
class AnomalyLocation:
    start: int
    end: int

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __json__(self):
        return {'start': self.start, 'end': self.end}

class IndependentAnomaly:
    dimension_id: int
    location: AnomalyLocation
    type: ANOMALY_TYPE

    def __init__(self):
        super().__init__()

    def __json__(self):
        return {'dimension_id': self.dimension_id, 'location': self.location.__json__(), 'type': self.type.name}


class CorrelatedAnomaly:
    dimension_ids: [int]
    location: AnomalyLocation
    type: ANOMALY_TYPE

    def __init__(self):
        super().__init__()

    def __json__(self):
        return {'dimension_ids': self.dimension_ids, 'location': self.location.__json__(), 'type': self.type.name}

class GENERATION_STRATEGY(Enum):
    ONLY_INDEPENDENT_SPIKE_ANOMALIES = 1
    ONLY_INDEPENDENT_DRIFT_ANOMALIES = 2
    ONLY_CORRELATED_SPIKE_ANOMALIES = 3
    ONLY_CORRELATED_DRIFT_ANOMALIES = 4
    MIX_INDEPENDENT_ANOMALIES = 5 # BOTH DRIFT AND SPIKE
    MIX_CORRELATED_ANOMALIES = 6 # BOTH CorrDrift AND CorrSpike
    MIX_INDEPENDENT_AND_CORRELATED_ANOMALIES = 5 # ALL 4 TYPES: Spike, Drift, CorrSpike, CorrDrift

class AnomalyGenerationAction:
    strategy: GENERATION_STRATEGY
    impacted_dimension_ids: [int]
    number_of_anomalies: int
    def __init__(self, strategy: GENERATION_STRATEGY, impacted_dimension_ids: [int], number_of_anomalies: int):
        self.strategy = strategy
        self.impacted_dimension_ids = impacted_dimension_ids
        self.number_of_anomalies = number_of_anomalies

    def __json__(self):
        return {'strategy': self.strategy.name,
                'impacted_dimension_ids': list(self.impacted_dimension_ids),
                'number_of_anomalies': self.number_of_anomalies}

class AnomalyGenerationActionMix(AnomalyGenerationAction):
    num_independent_spike_anomalies: int
    num_independent_spike_anomalies: int

    def __init__(self, strategy: GENERATION_STRATEGY, impacted_dimension_ids: [int], number_of_anomalies: int,
                 num_independent_spike_anomalies, num_independent_drift_anomalies):
        super().__init__(strategy, impacted_dimension_ids, number_of_anomalies)
        self.num_independent_spike_anomalies = num_independent_spike_anomalies
        self.num_independent_drift_anomalies = num_independent_drift_anomalies

    def __json__(self):
        base_json = super().__json__()
        base_json.update({
            'num_independent_spike_anomalies': self.num_independent_spike_anomalies,
            'num_independent_drift_anomalies': self.num_independent_drift_anomalies
        })
        return base_json

class AnomalyGenerationForASingleBatch:
    actions: [AnomalyGenerationAction]
    batch_df: pd.DataFrame

    def __init__(self, batch_df: pd.DataFrame, actions: [AnomalyGenerationAction]):
        super().__init__()
        self.batch_df = batch_df
        dimension_columns = [col for col in batch_df.columns if col.startswith('Sensor')]
        self.num_dimensions = batch_df[dimension_columns].shape[1]
        self.actions = [a for a in actions if all([d < self.batch_df.shape[1] for d in a.impacted_dimension_ids])]

    def __json__(self):
        return {'actions': [action.__json__() for action in self.actions], 'num_dimensions': self.num_dimensions}

    # def add_action(self, action: AnomalyGenerationAction):
    #     impacted_dimension_ids = action.impacted_dimension_ids
    #     for d in impacted_dimension_ids:
    #         assert d < self.num_dimensions, f'Impacted dimension id {d} exceeds number of dimensions {self.num_dimensions}'
    #     self.actions.append(action)
    # def add_actions(self, actions: [AnomalyGenerationAction]):
    #     for action in actions:
    #         impacted_dimension_ids = action.impacted_dimension_ids
    #         for d in impacted_dimension_ids:
    #             assert d < self.num_dimensions, f'Impacted dimension id {d} exceeds number of dimensions {self.num_dimensions}'
    #         self.actions.append(action)

class AnomalyGenerator:
    testing_dfs_dict: {str: pd.DataFrame}
    sensors_dict: dict[int, Sensor]

    def __init__(self, testing_dfs_dict: Dict[str, pd.DataFrame], sensors_dict: dict[int, Sensor]):
        self.testing_dfs_dict = testing_dfs_dict
        self.sensors_dict = sensors_dict
        # self.min_amp_list = min_amp_list
        # self.max_amp_list = max_amp_list
        # self.background_types = background_types
        # self.max_spike_length = max_spike_length

    def get_sensor_by_id(self, sensor_id: int) -> Sensor:
        return self.sensors_dict[sensor_id]

    def generate_anomalies_for_all_batches(self, actions_for_batches: {str: [AnomalyGenerationAction]}):
        for batch_id, actions in actions_for_batches.items():
            batch_df = self.testing_dfs_dict[batch_id]
            anomaly_generation_for_a_single_batch = AnomalyGenerationForASingleBatch(batch_df=batch_df, actions=actions)
            # anomaly_generation_for_a_single_batch.add_actions(actions)
            self.generate_anomalies_on_batch(batch_id, anomaly_generation_for_a_single_batch)
    def generate_anomalies_on_batch(self, batch_id: str, anomaly_generation_for_a_single_batch):

        new_df = self.execute_actions_on_df(anomaly_generation_for_a_single_batch)
        self.testing_dfs_dict[batch_id] = new_df
        print(f'Generated anomalies on batch {batch_id} with actions: {json.dumps(anomaly_generation_for_a_single_batch, default=lambda o: o.__json__() if hasattr(o, "__json__") else None, indent=4)}')

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
                    sensor: Sensor = self.get_sensor_by_id(sensor_index)
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
                            df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                            "Spike")
            elif action.strategy == GENERATION_STRATEGY.ONLY_INDEPENDENT_DRIFT_ANOMALIES:
                n_drifts = action.number_of_anomalies
                for sensor_index in action.impacted_dimension_ids:
                    sensor: Sensor = self.get_sensor_by_id(sensor_index)
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
                        drift = np.linspace(slope/2, slope, duration)
                        df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
                        df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                        "Drift")
            elif action.strategy == GENERATION_STRATEGY.ONLY_CORRELATED_SPIKE_ANOMALIES:
                impacted_dimension_ids = action.impacted_dimension_ids
                impacted_sensors = [self.get_sensor_by_id(sensor_index) for sensor_index in impacted_dimension_ids]
                num_anomalies = action.number_of_anomalies
                max_spike_length = int(min([s.spike_independent_anomaly_config.duration_range.max for s in impacted_sensors]))
                starts = pick_spike_starts_from(start, end, num_anomalies, max_spike_length, buffer=100)
                for st in starts:
                    ln = np.random.randint(1, max_spike_length + 1)
                    ed = min(st + ln - 1, end)
                    u = np.random.rand()
                    for sensor in impacted_sensors:
                        amp = sensor.get_min_amp_for_spike() + u * (sensor.get_max_amp_for_spike() - sensor.get_min_amp_for_spike())
                        # amp2 = min_amp2 + u * (max_amp2 - min_amp2)
                        sgn = np.random.choice([-1, 1])
                        df.loc[st - 1:ed - 1, f'Sensor{sensor.id}'] += sgn * amp
                        df[f'AnomalyFlag{sensor.id}'] = update_flags(df[f'AnomalyFlag{sensor.id}'],
                                                                        range(st, ed + 1),
                                                                        "SpikeCorr")

            elif action.strategy == GENERATION_STRATEGY.ONLY_CORRELATED_DRIFT_ANOMALIES:
                impacted_dimension_ids = action.impacted_dimension_ids
                impacted_sensors = [self.get_sensor_by_id(sensor_index) for sensor_index in impacted_dimension_ids]
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
            elif action.strategy == GENERATION_STRATEGY.MIX_INDEPENDENT_ANOMALIES: #min spike and drift in a dimension
                num_anomalies = action.number_of_anomalies
                num_spike_anomalies = np.random.randint(1, num_anomalies+1)
                num_drift_anomalies = num_anomalies - num_spike_anomalies

                for sensor_index in action.impacted_dimension_ids:
                    # self.inject_independent_spike_anomalies(df, sensor_index, start, end, max_independent_spikes, max_spike_length)
                    sensor: Sensor = self.get_sensor_by_id(sensor_index)
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
                            df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                            "Spike")

                for sensor_index in action.impacted_dimension_ids:
                    sensor: Sensor = self.get_sensor_by_id(sensor_index)
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
                        drift = np.linspace(slope/2, slope, duration)
                        df.loc[st - 1:ed - 1, f'Sensor{sensor_index}'] += drift
                        df[f'AnomalyFlag{sensor_index}'] = update_flags(df[f'AnomalyFlag{sensor_index}'], range(st, ed + 1),
                                                                        "Drift")

            elif action.strategy == GENERATION_STRATEGY.MIX_CORRELATED_ANOMALIES: #min Correlated spike and Correlated drift in a dimension
                impacted_dimension_ids = action.impacted_dimension_ids
                impacted_sensors = [self.get_sensor_by_id(sensor_index) for sensor_index in impacted_dimension_ids]
                num_anomalies = action.number_of_anomalies
                if num_anomalies == 1:
                    continue
                num_correlated_spike_anomalies = np.random.randint(1, num_anomalies)
                num_correlated_drift_anomalies = num_anomalies - num_correlated_spike_anomalies

                max_spike_length = int(
                    min([s.spike_independent_anomaly_config.duration_range.max for s in impacted_sensors]))
                starts = pick_spike_starts_from(start, end, num_correlated_spike_anomalies, max_spike_length, buffer=100)
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