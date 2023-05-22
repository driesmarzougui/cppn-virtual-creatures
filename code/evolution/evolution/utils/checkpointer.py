"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random
import time

import pickle

from neat.population import Population
from neat.reporting import BaseReporter, ReporterSet

from AESHN.shared.evolution_handler import EvolutionHandler


class Checkpointer(object):
    """
    A class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self, generation_interval=100, time_interval_seconds=300,
                 filename_prefix='neat-checkpoint-'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix

        self.current_generation = 0
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def check_checkpoint_due(self, generation: int) -> bool:
        self.current_generation = generation

        if self.generation_interval is not None:
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                return True 

        return False

    def checkpoint(self, config, population, species_set, archive_restoration, current_best):
        self.save_checkpoint(config, population, species_set, self.current_generation, archive_restoration,
                             current_best)
        self.last_generation_checkpoint = self.current_generation
        self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation, archive_restoration, current_best):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.filename_prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, random.getstate(), archive_restoration, current_best)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename, params, tb_logger, checkpointer, loggers):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, rndstate, archive_restoration, current_best = pickle.load(f)

            reporters = ReporterSet()
            for logger in loggers:
                reporters.add(logger)

            species_set = config.species_set_type(config.species_set_config, reporters)

            random.setstate(rndstate)
            return EvolutionHandler(config, params, tb_logger, checkpointer,
                                    checkpoint=(
                                    population, species_set, reporters, generation, archive_restoration, current_best))
