from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Dict

import numpy as np


class BACKGROUND_TYPE(Enum):
    AR_1_PROCESS = 1
    RANDOM_WALK = 2
    SINE_WAVE = 3
    POISSON_MOVING_AVERAGE = 4

class BackgroundSignal(ABC):
    @property
    @abstractmethod
    def background_type(self) -> BACKGROUND_TYPE:
        pass

    @abstractmethod
    def __json__(self):
        pass
class AR_1_Process_Signal(BackgroundSignal):
    background_phi: float

    def __init__(self, background_phi):
        self.background_phi = background_phi

    @property
    def background_type(self) -> BACKGROUND_TYPE:
        return BACKGROUND_TYPE.AR_1_PROCESS

    def __json__(self):
        return {
                'background_phi': self.background_phi,
                'background_type': self.background_type.name
                }

class Sine_Wave_Signal(BackgroundSignal):
    background_period: float
    background_amplitude: float

    def __init__(self, background_period, background_amplitude):
        self.background_period = background_period
        self.background_amplitude = background_amplitude

    @property
    def background_type(self) -> BACKGROUND_TYPE:
        return BACKGROUND_TYPE.SINE_WAVE

    def __json__(self):
        return {
                'background_period': self.background_period,
                'background_amplitude': self.background_amplitude,
                'background_type': self.background_type.name
                }

class Poisson_Moving_Average_Signal(BackgroundSignal):
    background_lambda: float
    background_k: int # Poisson Moving Window

    def __init__(self, background_lambda, background_k):
        self.background_lambda = background_lambda
        self.background_k = background_k

    @property
    def background_type(self) -> BACKGROUND_TYPE:
        return BACKGROUND_TYPE.POISSON_MOVING_AVERAGE

    def __json__(self):
        return {
                'background_lambda': self.background_lambda,
                'background_k': self.background_k,
                'background_type': self.background_type.name
                }
class Noise:
    mean: float
    sd: float

    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def __json__(self):
        return {'mean': self.mean, 'sd': self.sd}

class ValueRange:
    min: float
    max: float

    def __init__(self, min, max):
        self.min = min
        self.max = max

class ValueList:
    values: [float]

    def __init__(self, values):
        self.values = values

class AnomalyConfig(ABC):
    @property
    @abstractmethod
    def text(self):
        pass
class SpikeIndependentAnomalyConfig(AnomalyConfig):
    def __init__(self, magnitude_range: ValueRange, duration_range: ValueRange):
        self.magnitude_range = magnitude_range
        self.duration_range = duration_range

    def text(self):
        return 'Spike'
    def __json__(self):
        return {'magnitude_range': self.magnitude_range, 'duration_range': self.duration_range}
class DriftIndependentAnomalyConfig:
    def __init__(self, slope_range: ValueList, duration_range: ValueRange):
        self.slope_range = slope_range
        self.duration_range = duration_range

    def text(self):
        return 'Drift'
    def __json__(self):
        return {'slope_range': self.slope_range, 'duration_range': self.duration_range}

class Sensor:
    id: int
    noise: Noise
    spike_independent_anomaly_config: SpikeIndependentAnomalyConfig = None
    drift_independent_anomaly_config: DriftIndependentAnomalyConfig = None
    background_type: Union[BACKGROUND_TYPE | None] = None
    background_signal: Union[BackgroundSignal | None] = None

    def __init__(self, id: int, noise: Noise):
        self.id = id
        self.noise = noise

    def get_min_amp_for_spike(self):
        return max(self.noise.mean*self.noise.sd,self.noise.mean*self.spike_independent_anomaly_config.magnitude_range.min)
        # min_amp1, min_amp2 = 2 * sd1, 2 * sd2
        # max_amp1 = max(mean1 * spike_size, abs(mean1) * 1)
        # max_amp2 = max(mean2 * spike_size, abs(mean2) * 1)
        # return

    def get_max_amp_for_spike(self):
        return max(self.noise.mean * self.spike_independent_anomaly_config.magnitude_range.max, abs(self.noise.mean) * 1)

    def get_slope_range_for_drift(self):
        return self.drift_independent_anomaly_config.slope_range.values
    def get_duration_range_for_drift(self):
        return self.drift_independent_anomaly_config.duration_range

    def add_background_signal(self, background_signal: BackgroundSignal):
        self.background_type = background_signal.background_type
        self.background_signal = background_signal

    def set_spike_independent_anomaly_config(self, magnitude_range: ValueRange, duration_range: ValueRange):
        self.spike_independent_anomaly_config = SpikeIndependentAnomalyConfig(magnitude_range, duration_range)

    def set_drift_independent_anomaly_config(self, slope_range, duration_range):
        self.drift_independent_anomaly_config = DriftIndependentAnomalyConfig(slope_range, duration_range)

    @property
    def normal_noise_mean(self):
        return self.noise.mean

    @property
    def normal_noise_sd(self):
        return self.noise.sd
    def __json__(self):
        return {'id': self.id,
                # 'background_type': self.background_type.name,
                'background_signal': self.background_signal.__json__() if self.background_signal else None,
                'noise': self.noise.__json__(),
                'spike_independent_anomaly_config': self.spike_independent_anomaly_config.__json__() if self.spike_independent_anomaly_config else None,
                'drift_independent_anomaly_config': self.drift_independent_anomaly_config.__json__() if self.drift_independent_anomaly_config else None
                }


class CORRELATION_CONSTRAINT_TYPE(Enum):
    BACKGROUND_AR_1_CROSS_CORRELATION = 1
    BACKGROUND_RANDOM_WALK_CROSS_CORRELATION = 2
    NOISE_CROSS_CORRELATION = 3
class CorrelationConstraint(ABC):
    sensor_ids: [int]
    cross_correlation_matrix: Union[np.array, None] = None
    sensor_id_index_mapping: Union[Dict[int, int], None] = None
    correlation_type: CORRELATION_CONSTRAINT_TYPE

    def __init__(self, sensor_ids, sensor_id_index_mapping, cross_correlation_dict, correlation_type):
        for group_key in cross_correlation_dict.keys():
            id_1, id_2 = int(group_key.split('_')[-2]), int(group_key.split('_')[-1])
            assert id_1 in sensor_id_index_mapping.values() and id_2 in sensor_id_index_mapping.values(), f'Cross-correlation defined for sensors {id_1} and {id_2}, but only sensors {sensor_id_index_mapping.values()} are in the group.'
        self.sensor_ids = sensor_ids
        self.sensor_id_index_mapping = sensor_id_index_mapping
        self.cross_correlation_matrix = self.build_cross_correlation_matrix(cross_correlation_dict)
        self.correlation_type = correlation_type

    def build_sensor_id_index_mapping(self, sensor_ids):
        return {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}

    def build_cross_correlation_matrix(self, cross_correlation_dict) -> np.array:
        correlation_matrix = np.zeros((len(self.sensor_ids), len(self.sensor_ids)))
        for group_key, corr in cross_correlation_dict.items():
            index_1, index_2 = int(group_key.split('_')[-2]), int(group_key.split('_')[-1])
            # index_1 = sensor_id_index_mapping[sensor_id_1]
            # index_2 = sensor_id_index_mapping[sensor_id_2]
            correlation_matrix[index_1][index_2] = corr
            correlation_matrix[index_2][index_1] = corr
        return correlation_matrix

    def __str__(self):
        return (f'CorrelationConstraint(type={self.correlation_type.name}, '
                f'sensor_ids={self.sensor_ids}, '
                f'cross_correlation_matrix=\n{self.cross_correlation_matrix})')

class Sensor_Group(ABC):
    sensors: [Sensor]
    noise_cross_correlation: CorrelationConstraint

    def __init__(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint):
        self.sensors = sensors
        self.noise_cross_correlation = noise_cross_correlation

    @property
    @abstractmethod
    def background_type(self) -> BACKGROUND_TYPE:
        pass

#
# class AR_1_Sensor_Group(Sensor_Group):
#     ar_1_cross_correlation: CorrelationConstraint
#
#     def __init__(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint, ar_1_cross_correlation: CorrelationConstraint):
#         if self.validate(sensors, noise_cross_correlation, ar_1_cross_correlation):
#             super().__init__(sensors, noise_cross_correlation)
#             self.ar_1_cross_correlation = ar_1_cross_correlation
#         else:
#             raise ValueError('Invalid AR(1) sensor group configuration.')
#
#     @property
#     def background_type(self) -> BACKGROUND_TYPE:
#         return BACKGROUND_TYPE.AR_1_PROCESS
#
#     def validate(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint, ar_1_cross_correlation: CorrelationConstraint):
#         check1 = all([sensor.background_type == BACKGROUND_TYPE.AR_1_PROCESS for sensor in sensors])
#         check2 =  set(noise_cross_correlation.sensor_ids) == set(sensor.id for sensor in sensors)
#         check3 =  set(ar_1_cross_correlation.sensor_ids) == set(sensor.id for sensor in sensors)
#         return check1 and check2 and check3
#
#
# class Poisson_Moving_Average_Sensor_Group(Sensor_Group):
#
#     def __init__(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint):
#         if self.validate(sensors, noise_cross_correlation):
#             super().__init__(sensors, noise_cross_correlation)
#         else:
#             raise ValueError('Invalid Poisson moving average sensor group configuration.')
#     @property
#     def background_type(self) -> BACKGROUND_TYPE:
#         return BACKGROUND_TYPE.POISSON_MOVING_AVERAGE
#
#     def validate(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint):
#         check1 = all([sensor.background_type == BACKGROUND_TYPE.POISSON_MOVING_AVERAGE for sensor in sensors])
#         check2 =  set(noise_cross_correlation.sensor_ids) == set(sensor.id for sensor in sensors)
#         return check1 and check2
#
#
# class Sine_Wave_Sensor_Group(Sensor_Group):
#
#     def __init__(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint):
#         if self.validate(sensors, noise_cross_correlation):
#             super().__init__(sensors, noise_cross_correlation)
#         else:
#             raise ValueError('Invalid sine wave sensor group configuration.')
#
#     @property
#     def background_type(self) -> BACKGROUND_TYPE:
#         return BACKGROUND_TYPE.SINE_WAVE
#
#     def validate(self, sensors: [Sensor], noise_cross_correlation: CorrelationConstraint):
#         check1 = all([sensor.background_type == BACKGROUND_TYPE.SINE_WAVE for sensor in sensors])
#         check2 =  set(noise_cross_correlation.sensor_ids) == set(sensor.id for sensor in sensors)
#         return check1 and check2