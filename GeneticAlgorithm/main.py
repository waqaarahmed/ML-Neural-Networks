from random import choices, randint, randrange, random
from typing import List, Callable, Tuple
from collections import namedtuple

Genome = List[int]
Population = List[Genome]
Fitness_func = Callable[[Genome], int]
Populate_func = Callable[[], Population]
Selection_func = Callable[[Population, Fitness_func], Tuple[Genome, Genome]]
Crossover_Func = Callable[[Genome, Genome], Tuple[Genome, Genome]]
Mutation_Func = Callable[[Genome], Genome]
Thing = namedtuple('Thing', ['name', 'value', 'weight'])

things = [
    Thing('Laptop', '50000', '7'),
    Thing('Headphone', '1500', '1'),
    Thing('Mouse', '1500', '0.5'),
    Thing('DeskPad', '2000', '1'),
]
more_things = [
    Thing('Bubble', '10', '0.1'),
    Thing('Chocolate', '', '1'),
    Thing('Mouse', '1500', '0.5'),
    Thing('DeskPad', '2000', '1'),
]


def generate_genome(length):
    return choices([0, 1], k=length)


def generate_genome(size, genome_length):
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome, things, weight_limit):
    if len(genome) != len(things):
        raise ValueError("genome and things must be of same length")

    weight = 0
    value = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0

    return value


def selection_pair(population, fitness_func):
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )


def single_point_crossover(a, b):
    if len(a) != len(b):
        raise ValueError("Genome a and b must be of same length")
    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome, num, probability):
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
        return genome

