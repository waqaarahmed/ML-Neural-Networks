from random import choices
from typing import List

genome = List[int]
population = List[genome]

def generate_genome(length):
    return choices([0, 1], k=length)

def generate_genome(size, genome_length):
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome, things, weight_limit):
