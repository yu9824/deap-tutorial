from typing import (
    Optional,
    List,
    Dict,
    Any,
)
from operator import attrgetter

import numpy as np
import optuna

# from joblib import Parallel, delayed


# TODO
# すでに計算されている個体も存在するはずだがすべて再計算してしまっている（計算のロスが多い）
# [ ]: attrで世代を保存しておく`Dict[str, List[int]]`?

# FIXME: n_jobs = 1でないと正しく計算できない


class GeneticAlgorithmSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        n_gen: int = 100,
        popsize: int = 100,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
    ) -> None:
        self._rng = np.random.RandomState(seed)
        self.n_gen = n_gen
        self.popsize = popsize
        self.cxpb = cxpb
        self.mutpb = mutpb

        self._generation: int = 0
        self._populations: List[List[optuna.trial.FrozenTrial]] = []

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        self._trials_completed = study.get_trials(
            deepcopy=True, states=(optuna.trial.TrialState.COMPLETE,)
        )
        self._n_trials_completed = len(self._trials_completed)
        if self._n_trials_completed >= self.popsize:
            return self._trials_completed[0].distributions
        else:
            return {}

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: optuna.distributions.BaseDistribution,
    ) -> Dict[str, Any]:
        # FIXME: stateをwaitingなどに変更して、get_trialsで
        # completeなものはattrのgenerationに加えるだけでよく、
        # waitingのものはパラメータで返すようにすると計算量が削減できそう。

        # 世代が変わる瞬間
        if (
            self._n_trials_completed % self.popsize == 0
            and self._n_trials_completed > 0
        ):
            trials_parent = self._trials_completed[-self.popsize:]

            # select (tournament selection)
            tournsize = 3
            self._offspring: List[optuna.trial.FrozenTrial] = []
            for _ in range(self.popsize):
                aspirants = self._rng.choice(trials_parent, tournsize)
                if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                    self._offspring.append(
                        max(aspirants, key=attrgetter("value"))
                    )
                elif study.direction == optuna.study.StudyDirection.MINIMIZE:
                    self._offspring.append(
                        min(aspirants, key=attrgetter("value"))
                    )
                else:
                    raise ValueError(
                        "StudyDirection {} is not supported.".format(
                            study.direction
                        )
                    )

            # 偶数と奇数のペアで、一定の確率で交配 (crossover)
            for child1, child2 in zip(
                self._offspring[::2], self._offspring[1::2]
            ):
                size = min(len(child1.params), len(child2.params))
                cxpoint1 = self._rng.randint(1, size)
                cxpoint2 = (
                    self._rng.randint(1, size - 1) if size - 1 > 1 else 1
                )
                if cxpoint2 >= cxpoint1:
                    cxpoint2 += 1
                else:  # Swap the two cx points
                    cxpoint1, cxpoint2 = cxpoint2, cxpoint1
                lst_child1 = list(child1.params.items())
                lst_child2 = list(child2.params.items())
                (
                    lst_child1[cxpoint1:cxpoint2],
                    lst_child2[cxpoint1:cxpoint2],
                ) = (
                    lst_child2[cxpoint1:cxpoint2],
                    lst_child1[cxpoint1:cxpoint2],
                )
                child1.params = dict(lst_child1)
                child2.params = dict(lst_child2)

            # mutation
            indpb = 0.05
            for mutant in self._offspring:
                if self._rng.random() < self.mutpb:
                    # mutFlipBit
                    for key, value in mutant.params.items():
                        if self._rng.random() < indpb:
                            mutant.params[key] = not value

            # 保存
            self._populations.append(self._offspring)

            # 世代を更新
            self._generation += 1

        trial.set_user_attr("generation", self._generation)
        if self._generation > 0:
            return self._offspring.pop(0).params
        else:
            return {}

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        if isinstance(
            param_distribution, optuna.distributions.UniformDistribution
        ):
            generation = trial.user_attrs["generation"]
            if generation == 0:
                return self._rng.uniform(
                    param_distribution.low, param_distribution.high
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_uniform("x", -10, 10)
    y = trial.suggest_uniform("y", -10, 10)
    return (x - 2) ** 2 + (y + 5) ** 2


if __name__ == "__main__":
    n_gen = 100
    popsize = 100
    sampler = GeneticAlgorithmSampler(
        seed=334, n_gen=n_gen, popsize=popsize, cxpb=0.5, mutpb=0.2
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_gen * popsize)
