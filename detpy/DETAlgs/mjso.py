import copy
from typing import List

import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.crossover_methods.binomial_crossover import BinomialCrossover
from detpy.DETAlgs.data.alg_data import MJSOData
from detpy.DETAlgs.math.math_functions import MathFunctions
from detpy.DETAlgs.methods.methods_mjso import probability_selection
from detpy.DETAlgs.mutation_methods.current_to_pbest_r import MutationCurrentToPBestR
from detpy.DETAlgs.random.index_generator import IndexGenerator
from detpy.DETAlgs.random.random_value_generator import RandomValueGenerator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints_with_parent
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


class MJSO(BaseAlg):
    """
        MjSO: Modified jSO algorithm

        References:
        A novel modified jSO algorithm for enhanced numerical optimization.
        Swarm and Evolutionary Computation 78 (2023) 101294

        Key modifications over jSO/L-SHADE:
        1. Probability selection mechanism for x_pbest (better individuals have higher selection probability)
        2. Directed mutation strategy (x_r1 and x_r2 swapped based on fitness for better search direction)
        3. No external archive (x_r1 and x_r2 selected from current population only)


    """

    def __init__(self, params: MJSOData, db_conn=None, db_auto_write=False):
        super().__init__(MJSO.__name__, params, db_conn, db_auto_write)

        self._H = params.memory_size
        self._memory_F = np.full(self._H, 0.3)
        self._memory_Cr = np.full(self._H, 0.8)

        # Last element is fixed at 0.9 and never updated
        self._memory_F[self._H - 1] = 0.9
        self._memory_Cr[self._H - 1] = 0.9

        self._p_min = params.p_min
        self._p_max = params.p_max
        self._m = params.m

        self._k_index = 0

        self._success_Cr = []
        self._success_F = []
        self._difference_fitness_success = []

        self._min_pop_size = params.minimum_population_size
        self._start_population_size = self.population_size
        self._population_size_reduction_strategy = params.population_reduction_strategy

        self._TERMINAL = np.nan
        self._EPSILON = 0.00001

        self._index_gen = IndexGenerator()
        self._random_value_gen = RandomValueGenerator()
        self._binomial_crossing = BinomialCrossover()

    def _calculate_factors_for_epoch(self, pop_size: int):
        """
        Calculate F, Fw, and CR for each individual in the current epoch.

        F_i = Cauchy(M_{F,r3}, 0.1)
        Fw_i depends on FEs/FES_max ratio
        CR_i = Gaussian(M_{CR,r3}, 0.1) or 0 if terminal
        """
        f_table = []
        cr_table = []
        fw_table = []

        nfe_ratio = self.nfe / self.nfe_max

        for i in range(pop_size):
            ri = np.random.randint(0, self._H)

            # CR generation (Eq. 8)
            if np.isnan(self._memory_Cr[ri]):
                cr = 0.0
            else:
                cr = self._random_value_gen.generate_normal(self._memory_Cr[ri], 0.1, 0.0, 1.0)

            # F generation (Eq. 3)
            f = self._random_value_gen.generate_cauchy_greater_than_zero(self._memory_F[ri], 0.1, 0.0, 1.0)

            # Fw calculation (Eq. 4)
            if nfe_ratio < 0.2:
                fw = 0.7 * f
            elif nfe_ratio < 0.4:
                fw = 0.8 * f
            else:
                fw = 1.2 * f

            f_table.append(f)
            cr_table.append(cr)
            fw_table.append(fw)

        return f_table, cr_table, fw_table

    def _mutate(self, population: Population, f_table: List[float], fw_table: List[float]) -> Population:
        """
        Perform mutation with probability selection and directed mutation.

        N_pbest = NP * (p_min + (p_max - p_min) * FEs/FES_max)
        Probability selection for x_pbest
        Directed mutation with fitness-based x_r1/x_r2 ordering

        Uses MutationCurrentToPBestR: v = x_i + Fw*(x_pbest - x_i) + F*(r1 - r2)
        with r1/r2 pre-swapped based on fitness for directed mutation.
        """
        new_members = []
        nfe_ratio = self.nfe / self.nfe_max

        # Eq. (5): Calculate N_pbest
        n_pbest = int(round(population.size * (self._p_min + (self._p_max - self._p_min) * nfe_ratio)))
        n_pbest = max(2, min(n_pbest, population.size))

        for i in range(population.size):
            # Select x_pbest using probability selection (Eq. 15-16)
            pbest = probability_selection(population, n_pbest, self._m)

            # Select r1 and r2 from current population (no archive)
            idx_r1, idx_r2 = self._index_gen.generate_unique_multiple(population.size, 2, [i])
            member_r1 = population.members[idx_r1]
            member_r2 = population.members[idx_r2]

            # Directed mutation (Eq. 17-18): ensure r1 is the better individual
            if population.optimization == OptimizationType.MINIMIZATION:
                if member_r1.fitness_value > member_r2.fitness_value:
                    member_r1, member_r2 = member_r2, member_r1
            else:
                if member_r1.fitness_value < member_r2.fitness_value:
                    member_r1, member_r2 = member_r2, member_r1

            # v = x_i + Fw*(x_pbest - x_i) + F*(r1_better - r2_worse)
            mutated = MutationCurrentToPBestR.mutate(
                base_member=population.members[i],
                best_member=pbest,
                r1=member_r1,
                r2=member_r2,
                f=f_table[i],
                fw=fw_table[i]
            )

            new_members.append(mutated)

        return Population.with_new_members(population, new_members)

    def _selection(self, origin_population: Population, modified_population: Population,
                   f_table: List[float], cr_table: List[float]) -> Population:
        """
        Greedy selection with tracking of successful F and CR values.
        """
        optimization = origin_population.optimization
        new_members = []

        if optimization == OptimizationType.MINIMIZATION:
            is_better = lambda orig, mod: mod.fitness_value < orig.fitness_value
            diff = lambda orig, mod: orig.fitness_value - mod.fitness_value
        else:
            is_better = lambda orig, mod: mod.fitness_value > orig.fitness_value
            diff = lambda orig, mod: mod.fitness_value - orig.fitness_value

        for i in range(origin_population.size):
            orig = origin_population.members[i]
            mod = modified_population.members[i]

            if not is_better(orig, mod):
                new_members.append(copy.deepcopy(orig))
                continue

            self._success_F.append(f_table[i])
            self._success_Cr.append(cr_table[i])
            self._difference_fitness_success.append(diff(orig, mod))
            new_members.append(copy.deepcopy(mod))

        return Population.with_new_members(origin_population, new_members)

    def _update_memory(self, success_f: List[float], success_cr: List[float],
                       difference_fitness_success: List[float]):
        """
        Update memory M_F and M_CR using weighted Lehmer mean

        M_{F,k} = 0.5 * (mean_WL(S_F) + M_{F,k})
        M_{CR,k} with terminal logic
        mean_WL(S) = sum(w_n * S_n^2) / sum(w_n * S_n)
        w_n = |f(u_n) - f(x_n)| / sum|f(u_j) - f(x_j)|
        """
        if len(success_f) == 0 or len(success_cr) == 0:
            return

        # Don't update the last memory position (fixed at 0.9)
        if self._k_index >= self._H - 1:
            self._k_index = 0

        total = np.sum(difference_fitness_success)
        weights = np.array(difference_fitness_success) / total


        f_new = MathFunctions.calculate_lehmer_mean(np.array(success_f), weights, p=2)
        f_new = np.clip(f_new, 0, 1)
        self._memory_F[self._k_index] = 0.5 * (f_new + self._memory_F[self._k_index])


        if np.isclose(total, 0.0, atol=self._EPSILON):
            self._memory_Cr[self._k_index] = self._TERMINAL
        else:
            cr_new = MathFunctions.calculate_lehmer_mean(np.array(success_cr), weights, p=2)
            cr_new = np.clip(cr_new, 0, 1)
            self._memory_Cr[self._k_index] = 0.5 * (cr_new + self._memory_Cr[self._k_index])

        self._success_F = []
        self._success_Cr = []
        self._difference_fitness_success = []
        self._k_index = (self._k_index + 1) % (self._H - 1)

    def _update_population_size(self, nfe: int, total_nfe: int, start_pop_size: int, min_pop_size: int):
        """
        Linear Population Size Reduction.
        """
        new_size = self._population_size_reduction_strategy.get_new_population_size(
            nfe, total_nfe, start_pop_size, min_pop_size
        )
        self._pop.resize(new_size)

    def next_epoch(self):
        """
        Perform one generation of the MjSO algorithm.
        """
        f_table, cr_table, fw_table = self._calculate_factors_for_epoch(self._pop.size)

        # Mutation with probability selection and directed mutation
        mutant = self._mutate(self._pop, f_table, fw_table)

        # Binomial crossover
        trial = self._binomial_crossing.crossover_population(self._pop, mutant, cr_table)

        # Boundary constraint handling
        fix_boundary_constraints_with_parent(self._pop, trial, self.boundary_constraints_fun)

        # Evaluate trial vectors
        trial.update_fitness_values(self._function.eval, self.parallel_processing)

        # Greedy selection
        new_pop = self._selection(self._pop, trial, f_table, cr_table)

        # Update population
        self._pop = new_pop

        # Update memory for F and CR
        self._update_memory(self._success_F, self._success_Cr, self._difference_fitness_success)

        # Linear population size reduction
        self._update_population_size(self.nfe, self.nfe_max, self._start_population_size, self._min_pop_size)
