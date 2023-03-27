from typing import Optional, Union, Literal, Callable, Tuple, List
import warnings
from operator import attrgetter
from datetime import datetime

import numpy as np
from sklearn.utils import check_array, check_random_state
from deap import base
from deap import creator
from deap import tools

from joblib import Parallel, delayed


class GeneticAlgorithm:
    """遺伝的アルゴリズム(GA)は、適用範囲の非常に広い、生物の進化を模倣した 進化的アルゴリズムの一つである。
    自然界における生物の進化過程においては、ある世代(generation) を 形成している個体(individual)の集合、
    すなわち個体群(population)の中で、 環境への適合度(fitness)の高い個体が高い確率で生き残れるように
    再生(reproduction)される。さらに、交叉(crossover)や突然変異(mutation) によって、次の世代の個体群が形成されていく。

    https://www.gifu-nct.ac.jp/elec/deguchi/sotsuron/makino/node15.html
    """

    def __init__(
        self,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        direction: Literal["maximize", "minimize"] = "minimize",
        ipynb: bool = False,
    ) -> None:

        self.rng_ = check_random_state(random_state)

        self.direction = direction
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.direction == "maximize":
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            elif self.direction == "minimize":
                creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
            else:
                raise ValueError(
                    "direction must be either 'maximize' or 'minimize'"
                )

            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register(
            "create_ind",
            self.create_ind_uniform,
            vmins,
            vmaxs,
            random_state=self.rng_,
        )
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            toolbox.create_ind,
        )
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.individual
        )
        toolbox.register("evaluate", objective)
        # toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mate", self.cxTwoPoint, random_state=self.rng_)

        # toolbox.register(
        #     "mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2
        # )
        # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register(
            "mutate", self.mutFlipBit, indpb=0.05, random_state=self.rng_
        )

        # toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register(
            "select", self.selTournament, tournsize=3, random_state=self.rng_
        )

        """
        `toolbox.register(name, function, *args, **kwargs)`は、
        ```
        toolbox.__setattr__(
            name, lambda individuals: function(individuals, *args, **kwargs)
        )
        ```と考えればよさそう。
        """

        pop = toolbox.population(n=popsize)

        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        self.__populations: List[List] = []
        self.__fitnesses: List[List[float]] = []
        self.__time: List[Tuple[datetime, datetime]] = []
        for _ in self._range(n_gen):
            self.__populations.append(pop)
            _time_start = datetime.now()

            offspring = toolbox.select(pop, popsize)
            offspring = list(map(toolbox.clone, offspring))
            # 偶数と奇数のペアで、一定の確率で交配
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if self.rng_.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if self.rng_.random() < mutpb:
                    # オブジェクトをそのまま変えている
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # del してたものを再度評価し直す（突然変異or交叉をしているため）
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # populationを更新
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
            self.__fitnesses.append(fits)

            _time_end = datetime.now()
            self.__time.append((_time_start, _time_end))

        # 最も良い適用度の個体を取得
        self.best_ind = tools.selBest(pop, 1)[0]
        return self

    @property
    def populations(self):
        return self.__populations

    @property
    def fitnesses(self):
        return self.__fitnesses

    @property
    def time(self):
        return self.__time

    @populations.setter
    @fitnesses.setter
    @time.setter
    def _setter(self, value):
        raise AttributeError("can't set attribute")

    @staticmethod
    def create_ind_uniform(
        vmins,
        vmaxs,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ):
        rng_ = check_random_state(random_state)
        vmins = check_array(vmins, ensure_2d=False).flatten()
        vmaxs = check_array(vmaxs, ensure_2d=False).flatten()
        assert len(vmins) == len(
            vmaxs
        ), "vmins and vmaxs must be the same length"
        return [rng_.uniform(vmin, vmax) for vmin, vmax in zip(vmins, vmaxs)]

    @staticmethod
    def cxTwoPoint(
        ind1,
        ind2,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Tuple:
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
        cxpoint2 = rng_.randint(1, size - 1) if size - 1 > 1 else 1
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
    ) -> Tuple:
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
        individuals,
        k: int,
        tournsize: int,
        fit_attr="fitness",
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> List:
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
