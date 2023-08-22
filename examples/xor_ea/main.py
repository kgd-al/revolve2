"""
Run the example.

Optimizes a neural network for solving XOR using a simple evolutionary algorithm.
"""

import hashlib
import logging
from typing import List, Tuple, cast

import config
import numpy as np
from base import Base
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population
from revolve2.core.database import OpenMethod, open_database_sqlite
from revolve2.core.optimization.ea import population_management, selection
from sqlalchemy.orm import Session

ParamTuple = Tuple[float, float, float, float, float, float, float, float, float]


def select_parents(
    rng: np.random.Generator,
    population: Population,
    offspring_size: int,
) -> List[Tuple[int, int]]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents.
    """
    return cast(
        List[Tuple[int, int]],
        [
            selection.multiple_unique(
                2,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
            )
            for _ in range(offspring_size)
        ],
    )


def select_survivors(
    rng: np.random.Generator,
    original_population: Population,
    offspring_population: Population,
) -> Population:
    """
    Select survivors using a tournament.

    :param rng: Random number generator.
    :param original_population: The population the parents come from.
    :param offspring_population: The offspring.
    :returns: A newly created population.
    """
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return Population(
        [
            Individual(
                original_population.individuals[i].genotype,
                original_population.individuals[i].fitness,
            )
            for i in original_survivors
        ]
        + [
            Individual(
                offspring_population.individuals[i].genotype,
                offspring_population.individuals[i].fitness,
            )
            for i in offspring_survivors
        ]
    )


def relu(val: float) -> float:
    """
    Calculate relu for the given number.

    :param val: The value to calculate for.
    :returns: The calculated value.
    """
    return max(0, val)


def evaluate_network(params: ParamTuple, input1: float, input2: float) -> float:
    """
    Pass two inputs through a fully connected relu network.

    :param params: The parameters to evaluate.
    :param input1: First input for network.
    :param input2: Second input for network.
    :returns: The output of the network.
    """
    # usually you would do this with matrix multiplications and numpy,
    # but leaving it manualy for clarity
    n0 = relu(input1 * params[0] + input2 * params[1] + params[2])
    n1 = relu(input1 * params[3] + input2 * params[4] + params[5])
    return relu(n0 * params[6] + n1 * params[7] + params[8])


def evaluate(parameters: ParamTuple) -> Tuple[float, float, float, float, float]:
    """
    Measure one set of parameters.

    :param parameters: The parameters to measure.
    :returns: Sum of squared errors and each individual error.
    """
    ios = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]

    results = [evaluate_network(parameters, io[0], io[1]) for io in ios]
    errors = tuple(float(abs(result - io[2])) for result, io in zip(results, ios))

    return cast(
        Tuple[float, float, float, float, float],
        (sum([-(err**2) for err in errors]),) + errors,
    )


def main() -> None:
    """Run the program."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # create a unique seed and initialize the random number generator
    rng_seed = int(
        hashlib.sha256(f"xor_ea_seed{config.RNG_SEED}".encode()).hexdigest(),
        16,
    )
    rng = np.random.Generator(np.random.PCG64(rng_seed))

    # open the database
    dbengine = open_database_sqlite(
        config.DATABASE_FILE,
        open_method=OpenMethod.NOT_EXISTS_AND_CREATE,
    )
    Base.metadata.create_all(dbengine)

    # create initial population and evaluate
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]
    logging.info("Evaluating initial population.")
    initial_fitnesses = [
        evaluate(cast(ParamTuple, genotype.parameters))[0]
        for genotype in initial_genotypes
    ]
    population = Population(
        [
            Individual(genotype, fitness)
            for genotype, fitness in zip(initial_genotypes, initial_fitnesses)
        ]
    )

    # create initial generation from initial population
    generation = Generation(
        0,
        population,
    )

    # save the initial generation
    logging.info("Saving initial population.")
    with Session(dbengine, expire_on_commit=False) as ses:
        ses.add(generation)
        ses.commit()

    # below is the actual optimization process
    logging.info("Start optimization process.")
    while generation.generation_index < config.NUM_GENERATIONS:
        logging.info(
            f"Generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
        )

        # create offspring
        parents = select_parents(rng, generation.population, config.OFFSPRING_SIZE)
        offspring_genotypes = [
            Genotype.crossover(
                generation.population.individuals[parent1_i].genotype,
                generation.population.individuals[parent2_i].genotype,
                rng,
            ).mutate(rng)
            for parent1_i, parent2_i in parents
        ]

        # evaluate the offspring
        offspring_fitnesses = [
            evaluate(cast(ParamTuple, genotype.parameters))[0]
            for genotype in offspring_genotypes
        ]

        # make an intermediate offspring population
        offspring_population = Population(
            [
                Individual(genotype, fitness)
                for genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)
            ]
        )

        # create the next population by selecting survivors
        survived_population = select_survivors(
            rng,
            generation.population,
            offspring_population,
        )

        # make it into then next generation
        generation = Generation(
            generation.generation_index + 1,
            survived_population,
        )

        # save the newly created generation
        with Session(dbengine, expire_on_commit=False) as ses:
            ses.add(generation)
            ses.commit()


if __name__ == "__main__":
    main()