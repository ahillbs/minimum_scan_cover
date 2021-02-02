import numpy as np

def flip_mutation(genomes: np.array, genome_mutation_chance=0.03, dna_mutation_chance=0.03):
    genomes_size = genomes.shape[0]
    dna_size = len(genomes[0])
    # random choice of a genome via random indices
    genomes_chance = np.random.random(genomes_size)
    mutation_indices = np.where(genomes_chance < genome_mutation_chance)[0]
    for index in mutation_indices:
        genome = genomes[index]
        dna_chances = np.random.random(dna_size)
        mutated_dna = np.where(dna_chances < dna_mutation_chance)[0]
        genome.feature_mask[mutated_dna] = 1 - genome.feature_mask[mutated_dna]
    # Repair genomes, if every entry in the mask has less than two features
    at_least_two_features(genomes)

def at_least_two_features(genomes):
    for genome in genomes:
        mask = genome.feature_mask
        if mask.sum() < 2:
            zero_mask = mask[mask == 0]
            # The amount of entries, that needs to change so at least two are 1
            to_change = 2 - (len(mask) - len(zero_mask))
            shuffled = np.arange(len(zero_mask))
            np.random.shuffle(shuffled)
            zero_mask[shuffled[:to_change]] = 1
            mask[mask == 0] = zero_mask