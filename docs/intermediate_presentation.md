# NeuroMorpholution
"Natural Evolution of Agent Brain and Morphology in a Virtual Ecosystem"


## Contents
* Problem definition
* Goals & motivation
* Related work
* Methodology
* Future work


## Problem definition (4 min)
"How can evolution be used as a tool to create intelligent agents with advantageous morphologies"

-> evolution
-> intelligent agents
-> advantageous morphologies

### A quick primer on Evolution
* Short explanation of evolutionary principles using e.g. dogs / horses
  * Describe mutation, reproduction, selection, genotype (genome), phenotype, offspring
* Refer to evolutionary (and mainly genetic) algorithms


### Intelligent agents
* Artificial intelligence
* Short description of intelligent agents
  * Often: intelligence interpretable as the ability to (learning) to do the given task given certain inputs
* Short Neural Net primer -> and how they can be evolved


### Advantageous morphologies
* A well working morphology is important for a well working agent
* Human thumbs
* Tight link between brain and morphology is important -> embodied intelligence


### Evolution as the underlying process for intelligence
* Explain "using the principles of evolution to create intelligent agents with advantageous morphologies" using the previously described context

## Motivation - why evolution (1 min)
* "Why evolution and not just e.g. reinforcement learning?"
* Our world
* Design the process, not the result
* Limit human design part (brain and morphology design) -> open endedness -> virtual ecosystem


## Goals (1 min)
* Refer first slide title
* Studying the dynamics between simultaneously evolving brains and morphologies
  * How the body shapes the mind and how the mind shapes the body
  * The effect of this paradigm within individual agent optimisation in a certain task (here: natural survival)
  * Designing new techniques to do this
* Studying the inter-agent interactions (cooperative and competititve behaviors) of agents with evolving brains and morphologies
* Approximating open-endedness in a virtual ecosystem through approximation of natural processes
* Creating a tool / framework to conduct these kind of evolution experiments

## Related work (3 min)

* Morphological evolution
  * Evolving 3D Morphology and Behavior by competition - Karl Sims (todo: add year)
    * show video
  * Framsticks
  * Robogrammar
  * Soft robots CPPN

* NeuroEvolution
  * Polyworld - Yaegar (todo: add year)
    * Artificial life
    * Introduces a computer model of living organisms and the ecology they exist in called polyworld
    * Brings together biologically motivated genetics, simple simulated physiologies and metabolisms. Hebbian learning in arbitrary neural network architectures, a visual perceptive mechanism and primitive behaviors in artificial organisms in an ecology just complex enough to foster speciation and inter-species competition 
    * Evolves brains but not the morphology

  * NEAT -> HyperNEAT -> ES-HyperNEAT -> Adaptive ES-HyperNEAT
    * Explain CPPN & substrate -> geometry
    * Complexification
    * (Speciation)

* Simultaneous evolution of body and brain:
  * Hasn't been done a lot, often a body is evolved and the control is learned through RL or other techniques
  * Only one real example found: Embodied embeddings for HyperNEAT


## Methodology (4 min)

### Virtual ecosystem
* Unity environment with pictures / video
* Agent observations & actions
* Agent life & metabolism

### Simultaneously evolving brain and body
* DNA -> "one CPPN to rule them all" (Novelty)
* Morphology phenotype
* Brain phenotype


### The search (leave out if not enough time) 
* Fitness
* Novelty search


## Current status (1 min)
* Refer the unity simulation shown previously
* Conducting ES-HyperNEAT experiments on virtual node wall ~ basic agent survival

## Future work (1 min)
* Finish basic survival experiments (DEC)
* Enable reproduction -> SBS experiments -> NeuroEvolution part finished (JAN)
* Enable morphological evolution (FEB / MARCH)
  * Create new world with e.g. mountains to make morphology more important

