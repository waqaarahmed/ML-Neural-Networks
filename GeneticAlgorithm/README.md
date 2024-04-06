Genetic Algorithm for Predicting Supply Chain Shortages
This code implements a genetic algorithm framework to predict supply chain shortages based on a given set of supply chain data. The genetic algorithm evolves populations of genomes to minimize the shortage cost, which is calculated based on the selected products in each genome.
Modules:
- random:  Python module for generating random numbers and choices.
- typing:  Python module for type hints.
Types:
- Genome:  A list of integers representing a genome, where each integer corresponds to a binary decision for selecting a product in the supply chain data.
- Population:  A list of genomes representing a population.
- PopulateFunc:  A function type that generates a population.
- FitnessFunc:  A function type that computes the fitness of a genome.
- SelectionFunc:  A function type that selects genomes for reproduction.
- CrossoverFunc:  A function type that performs crossover between parent genomes.
- MutationFunc:  A function type that performs mutation on a genome.
- PrinterFunc:	  A function type that prints population statistics.
Functions:
1. generate_genome() -> Genome:
   - Generates a random genome representing the selection of products.
2. calculate_shortage_cost(genome: Genome) -> int:
   - Calculates the total shortage cost based on the selected products in the genome.
3. generate_population(size: int) -> Population:
   - Generates a population of genomes of the specified size.
4. single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
   - Performs single-point crossover between two parent genomes.
5. mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
   - Performs mutation on a genome by flipping random bits.
6. population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
   - Computes the total fitness of a population based on the fitness function.
7. `selection_pair(population: Population, fitness_func: FitnessFunc) -> Population`:
   - Selects pairs of parent genomes for reproduction based on their fitness.
8. sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
   - Sorts a population based on the fitness of its genomes.
9. genome_to_string(genome: Genome) -> str:
   - Converts a genome to a string representation.
10. print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    - Prints statistics for the population, including the total shortage cost and the best genome.
11. run_evolution(...):
    - Runs the genetic algorithm to evolve populations of genomes and minimize the shortage cost.
Usage:
1. Define the supply chain data.
2. Run the genetic algorithm by calling `run_evolution()` with appropriate parameters.
3. Monitor the evolution process and analyze the results.
Example:
See the example provided in the code for how to use the genetic algorithm framework to predict supply chain shortages.
Note:
- This code is a simplified implementation and may require adjustments based on specific use cases or requirements.
- Ensure that the supply chain data format and the fitness function are appropriate for the problem domain.
