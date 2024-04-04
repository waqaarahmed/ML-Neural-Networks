from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple

# Define types
Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

# Supply chain data (example)
supply_chain_data = [
    # Product ID, Demand, Supply, Shortage Cost
    [1, 100, 80, 10],
    [2, 120, 110, 8],
    [3, 150, 140, 5],
    [4, 90, 100, 12],
    [5, 80, 70, 7]
]

def generate_genome() -> Genome:
    return choices([0, 1], k=len(supply_chain_data))

def calculate_shortage_cost(genome: Genome) -> int:
    # Calculate the total shortage cost based on selected products
    total_shortage_cost = 0
    for i, bit in enumerate(genome):
        if bit == 1:
            demand, supply, shortage_cost = supply_chain_data[i][1:]
            if supply < demand:
                shortage_quantity = demand - supply
                total_shortage_cost += shortage_quantity * shortage_cost
    return -total_shortage_cost  # Negative to maximize fulfillment

def generate_population(size: int) -> Population:
    return [generate_genome() for _ in range(size)]

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome

def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )

def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)

def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))

def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Total shortage cost: %d" % population_fitness(population, fitness_func))
    print("Best genome: %s" % genome_to_string(population[0]))
    print("")

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i

# Run the genetic algorithm
population_size = 100

population, generations = run_evolution(
    populate_func=lambda: generate_population(population_size),
    fitness_func=calculate_shortage_cost,
    fitness_limit=0,  # You can set a target shortage cost to stop the algorithm
    printer=print_stats
)
