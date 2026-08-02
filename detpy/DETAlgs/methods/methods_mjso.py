import numpy as np

from detpy.models.member import Member
from detpy.models.population import Population


def probability_selection(population: Population, n_pbest: int, m: float) -> Member:
    """
    Select x_pbest from the top N_pbest individuals using probability selection mechanism.
    Better individuals have higher probability of being selected.

    Pr_i = R_i / sum(R_j)
    ER_i = m - (m-1)*(i-1) / (N_pbest - 1)

    Parameters:
    - population (Population): Current population.
    - n_pbest (int): Number of top individuals to consider.
    - m (float): Probability scale factor controlling ratio between best and worst probabilities.

    Returns:
    - Member: The selected x_pbest individual.
    """
    n_pbest = max(2, n_pbest)
    best_members = population.get_best_members(n_pbest)

    if n_pbest == 1:
        return best_members[0]

    r_values = np.array([m - (m - 1) * i / (n_pbest - 1) for i in range(n_pbest)])
    probabilities = r_values / np.sum(r_values)

    selected_index = np.random.choice(n_pbest, p=probabilities)
    return best_members[selected_index]
