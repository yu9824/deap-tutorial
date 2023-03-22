from typing import Optional, Union, Literal, Callable, Tuple, List, Generator
from copy import deepcopy
from operator import attrgetter

import numpy as np
from sklearn.utils import check_array, check_random_state

from joblib import Parallel, delayed

# TODO: deapを使わずに書き直す
# TODO: trialを用いた形に変更してstudy-likeに使えるようにする


class Individual(list):
    def __init__(self, *args, sign=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sign = sign
        if self.sign not in [-1, 1]:
            raise ValueError("sign must be -1 or 1")
        self._fitness = None

    @property
    def fitness(self) -> Optional[Tuple[float]]:
        return (
            self._fitness
            if self._fitness is None
            else self.sign * self._fitness
        )

    @fitness.setter
    def fitness(self, value):
        self._fitness = value


class GeneticAlgorithm:
    """遺伝的アルゴリズム(GA)は、適用範囲の非常に広い、生物の進化を模倣した 進化的アルゴリズムの一つである。
    自然界における生物の進化過程においては、ある世代(generation) を 形成している個体(individual)の集合、
    すなわち個体群(population)の中で、 環境への適合度(fitness)の高い個体が高い確率で生き残れるように
    再生(reproduction)される。さらに、交叉(crossover)や突然変異(mutation) によって、次の世代の個体群が形成されていく。

    https://www.gifu-nct.ac.jp/elec/deguchi/sotsuron/makino/node15.html
    """

    def __init__(
        self,
        direction: Literal["maximize", "minimize"] = "minimize",
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        ipynb: bool = False,
    ) -> None:

        self.direction = direction
        self.rng_ = check_random_state(random_state)
        self.ipynb = ipynb

        try:
            if self.ipynb:
                from tqdm.notebook import trange

                self._range = trange
            else:
                from tqdm import trange

                self._range = trange
        except ImportError:
            self._range = range

    def optimize(
        self,
        objective: Callable[[np.ndarray], Tuple[float]],
        vmins,
        vmaxs,
        n_gen: int = 100,
        popsize: int = 100,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
    ):

        if self.direction == "maximize":
            sign = 1
        elif self.direction == "minimize":
            sign = -1
        else:
            raise ValueError("direction must be maximize or minimize")

        population: List[Individual] = [
            Individual(
                self.create_ind_uniform(vmins, vmaxs, random_state=self.rng_),
                sign=sign,
            )
            for _ in range(popsize)
        ]

        _fitnesses: Generator[Tuple[float]] = map(objective, population)
        for ind, fit in zip(population, _fitnesses):
            ind.fitness = fit

        for i_generation in self._range(n_gen):
            offspring = self.selTournament(
                individuals=population, k=popsize, tournsize=3
            )
            offspring = list(map(deepcopy, offspring))
            # 偶数と奇数のペアで、一定の確率で交配
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if self.rng_.random() < cxpb:
                    self.cxTwoPoint(child1, child2, random_state=self.rng_)
                    child1.fitness = None
                    child2.fitness = None

            for mutant in offspring:
                if self.rng_.random() < mutpb:
                    # オブジェクトをそのまま変えている
                    self.mutFlipBit(mutant, indpb=0.05)
                    mutant.fitness = None

            invalid_ind = [ind for ind in offspring if not ind.fitness]
            # del してたものを再度評価し直す（突然変異or交叉をしているため）
            _fitnesses: Generator[Tuple[float]] = map(objective, invalid_ind)
            for ind, fit in zip(invalid_ind, _fitnesses):
                ind.fitness = fit

            # populationを更新
            population[:] = offspring
            # fits = [ind.fitness.values[0] for ind in pop]
            # 保存するならば、各世代のpopulationとfitness。

        # 最も良い適用度の個体を取得
        self.best_ind = sorted(
            population, key=attrgetter("fitness"), reverse=True
        )[0]
        return self

    @staticmethod
    def create_ind_uniform(
        vmins,
        vmaxs,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> List[float]:
        rng_ = check_random_state(random_state)
        vmins = check_array(vmins, ensure_2d=False).flatten()
        vmaxs = check_array(vmaxs, ensure_2d=False).flatten()
        assert len(vmins) == len(
            vmaxs
        ), "vmins and vmaxs must be the same length"
        return [rng_.uniform(vmin, vmax) for vmin, vmax in zip(vmins, vmaxs)]

    @staticmethod
    def cxTwoPoint(
        ind1: Individual,
        ind2: Individual,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Tuple[Individual, Individual]:
        """Executes a two-point crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.

        Parameters
        ----------
        ind1 : List[individual]
             The first individual participating in the crossover.
        ind2 : List[individual]
            The second individual participating in the crossover.
        random_state : Optional[Union[np.random.RandomState, int]], optional
            seed, by default None

        Returns
        -------
        Tuple
            A tuple of two individuals.
        """
        rng_ = check_random_state(random_state)

        size = min(len(ind1), len(ind2))
        cxpoint1 = rng_.randint(1, size)
        cxpoint2 = rng_.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
            ind2[cxpoint1:cxpoint2],
            ind1[cxpoint1:cxpoint2],
        )

        return ind1, ind2

    @staticmethod
    def mutFlipBit(
        individual,
        indpb: float,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Tuple[Individual]:
        """Flip the value of the attributes of the input individual and return
        the mutant. The *individual* is expected to be a :term:`sequence` and
        the values of the attributes shall stay valid after the ``not``
        operator is called on them. The *indpb* argument is the probability of
        each attribute to be flipped.
        This mutation is usually applied on boolean individuals.


        Parameters
        ----------
        individual : _type_
            Individual to be mutated.
        indpb : float
            Independent probability for each attribute to be flipped.
        random_state : Optional[Union[np.random.RandomState, int]], optional
            _description_, by default None

        Returns
        -------
        Tuple
            A tuple of one individual.
        """
        rng_ = check_random_state(random_state)
        for i in range(len(individual)):
            if rng_.random() < indpb:
                individual[i] = type(individual[i])(not individual[i])

        return (individual,)

    @staticmethod
    def selTournament(
        individuals: List[Individual],
        k: int,
        tournsize: int,
        fit_attr="fitness",
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> List[Individual]:
        """Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.

        Parameters
        ----------
        individuals : List[individual]
            A list of individuals to select from.
        k : int
            The number of individuals to select.
        tournsize : int
            The number of individuals participating in each tournament.
        fit_attr : str, optional
            The attribute of individuals to use as selection criterion
            , by default "fitness"
        random_state : Optional[Union[np.random.RandomState, int]], optional
            _description_, by default None

        Returns
        -------
        List[individual]
            A list of selected individuals.
        """
        rng_ = check_random_state(random_state)

        chosen = []
        for i in range(k):
            aspirants = [
                individuals[i]
                for i in rng_.choice(len(individuals), tournsize)
            ]
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen


if __name__ == "__main__":

    def objective(x):
        return (np.sum(x),)

    def recursive(i):
        ga = GeneticAlgorithm(random_state=i)
        ga.optimize(objective, vmins=[-1, -1, -1], vmaxs=[1, 1, 1])

    Parallel(n_jobs=-1)([delayed(recursive)(i) for i in range(10)])
    # recursive(1)
