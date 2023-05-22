from typing import Dict

import neat

from AESHN.shared import bss


def create_config(params: Dict) -> neat.Config:
    """
    Loads a NEAT config from the given experiment parameter file.
    """
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                       neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                       params["experiment"]["config_path"])

    return config
