import itertools
import random

import numpy as np

from anomaly_generator import AnomalyGenerationAction, GENERATION_STRATEGY
from entities.sensors.data_model import CorrelationConstraint, CORRELATION_CONSTRAINT_TYPE


class ActionGenerator():
    n_sensors: int
    correlated_constraints: list[CorrelationConstraint]
    def __init__(self, n_sensors: int, correlated_constraints: list[CorrelationConstraint]):
        self.n_sensors = n_sensors
        self.correlated_constraints = correlated_constraints

    def generate_independent_actions_for_single_dimension(self) -> list[list[AnomalyGenerationAction]]:
        strategies = [
            GENERATION_STRATEGY.ONLY_INDEPENDENT_SPIKE_ANOMALIES,
            GENERATION_STRATEGY.ONLY_INDEPENDENT_DRIFT_ANOMALIES,
            GENERATION_STRATEGY.MIX_INDEPENDENT_ANOMALIES,
        ]
        number_of_anomalies_options = [3]
        impacted_dimension_ids_options = list([[i] for i in range(self.n_sensors)])
        impacted_dimension_ids_options.extend([list(p) for p in itertools.combinations(range(self.n_sensors), 2)])
        # impacted_dimension_ids_options = list(itertools.chain(*[itertools.combinations(range(self.n_sensors), ni) for ni in range(1, min(4, self.n_sensors))]))

        results = []
        for strategy, impacted_dimension_ids, number_of_anomalies in itertools.product(strategies, impacted_dimension_ids_options, number_of_anomalies_options):
            new_action = AnomalyGenerationAction(strategy=strategy,
                                    impacted_dimension_ids=impacted_dimension_ids,
                                    number_of_anomalies=number_of_anomalies
                                    )
            results.append([new_action])
        return results

    def generate_correlated_actions_for_multiple_dimension(self)-> list[list[AnomalyGenerationAction]]:
        correlated_strategies = [
            GENERATION_STRATEGY.ONLY_CORRELATED_SPIKE_ANOMALIES,
            GENERATION_STRATEGY.ONLY_CORRELATED_DRIFT_ANOMALIES,
            GENERATION_STRATEGY.MIX_CORRELATED_ANOMALIES
        ]

        number_of_anomalies_list = [3]

        impacted_dimension_correlated_ids_list = [correlated_constraint.sensor_ids for correlated_constraint in
                                                  self.correlated_constraints if correlated_constraint.correlation_type.value == CORRELATION_CONSTRAINT_TYPE.NOISE_CROSS_CORRELATION.value]

        # for c in self.correlated_constraints:
        #     print('Correlation_type', c.correlation_type, type(c.correlation_type))
        #     if c.correlation_type is CORRELATION_CONSTRAINT_TYPE.NOISE_CROSS_CORRELATION:
        #         print(f'Positive correlation constraint between sensors: {c.sensor_ids}')

        # configurations_correlated = list(
        #     itertools.product(correlated_strategies, impacted_dimension_correlated_ids_list,
        #                       number_of_anomalies_list[1:]))
        results = []
        for strategy, impacted_dimension_ids, number_of_anomalies in itertools.product(correlated_strategies,
                                                                                       impacted_dimension_correlated_ids_list,
                                                                                       number_of_anomalies_list):
            new_action = AnomalyGenerationAction(strategy=strategy,
                                                 impacted_dimension_ids=impacted_dimension_ids,
                                                 number_of_anomalies=number_of_anomalies
                                                 )
            results.append([new_action])
        return results

    def generate_mix_actions(self)-> dict:
        independent_actions = self.generate_independent_actions_for_single_dimension()
        correlated_actions = self.generate_correlated_actions_for_multiple_dimension()

        # mix_independent_actions = random.sample([list(p) for p in itertools.combinations(independent_actions, 3)], k=min(20, len(independent_actions)))
        mix_correlated_actions = random.sample([list(itertools.chain(*list(p))) for p in list(itertools.combinations(correlated_actions, 2))], k=min(20, len(correlated_actions)))
        mix_actions = [p[0] + p[1] for p in list(itertools.product(independent_actions, mix_correlated_actions))]
        # mix_actions = [list(p) for p in itertools.combinations(mix_actions, 2)]
        all_actions = random.sample(mix_actions, k=min(500, len(mix_actions)))
        # mixed_actions = []
        # for action_combination in itertools.combinations(all_actions, num_executed_action):
        #     mixed_actions.append(list(action_combination))
        return dict({
            'action_on_single_dimension': independent_actions,
            'action_on_single_correlated_group': correlated_actions,
            'mixed_actions': all_actions
        })
