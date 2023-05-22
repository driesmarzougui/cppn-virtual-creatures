# Evolving Virtual Creatures Through Geometrical Patterns

## Intro

* Imagine a life without a thumb
  * Examples of things that a thumb is handy for
  * The typical example of evolution
* Evolution gave us thumbs, so why not AI
  * Has shown the ability to simultaneously create intelligent creatures with smart morphologies
    * Really interesting for robotics!
  * Intuitive reasons behind the evolutionary approach
    * Designing the processes that creates intelligence instead of directly trying to design intelligent agents
      * This process (evolution) seems easier / interesting road to follow
    * Limiting the part that humans take in the design of AI
      * Evolution has a higher creative potential
      * Open endedness

Goal: grab attention & introduce evolution based AI

## [MAYBE] Just imagine
* You have some (labour intensive) task & you want a robot to do it for you
* Just evolve one!

## The objective of this thesis
* Just like evolution in nature, use evolution as a tool to create intelligent agents with fitting morphologies


## Evolutionary computation

* Idea of using evolution for optimization ofcourse isn't new
* Many types of EAs exist
* Focus on genetic algorithms
  * Genotype & phenotype & genotype to phenotype mapping
  * the typical select-mutate/crossover-eval loop

Goal: briefly go more into detail about evolutionary algorithms, focusing on genetic algorithms
  and explain terminology such as "genotype", "phenotype"

## Virtual Creatures - intro 
* Introduce virtual creatures
  * Agents with a brain and morphology
  * Perfect to study brain-morph evolution with
  * Often studied within the artificial life domain 
  * Often studied using the evolutionary techniques described earlier
* Note: outside of the more philosophical artificial life domain, e.g. in the pure robotics domain this type of research of course also has valuable implications

* Show some example virtual creatures
* Preview: Add an example of a crawler (evolved in this thesis)

## Evolving Virtual Creatures
* Cool examples from literature (show with underlying genome)
  * Morphological evolution
    * K. Sims & Framsticks & Robogrammar & Soft bodies
  * NeuroEvolution - evolving brains
    * Controller often created by e.g. RL; neuroevolution is the evolutionary alternative
    * NEAT example

## Virtual Ecosystem
* Virtual creatures need some environment in which they evolve --> natural choice is a Virtual Ecosystem > polyworld
  * why?

Goal: introduce virtual creatures and show cool examples from literature
  thereby also introducing morphological evolution, neuroevolution and virtual ecosystem

## The current problem with simultaneous evolution of brain and body
* SEMC
  * Virtual creatures require both a brain and body
    * Interesting subject to study simultaneous brain body optimization on
  * Current issue: uncomplementary changes brain-body
    * Thats why often optimized seperately or iteratively
  * Simultaneous optimization nevertheless has benefits
* Addressing this issue is the main research objective of this thesis

## Geometrical patterns as the solution: The CPPN

* CPPN and geometrical patterns intro

* We can use a neural network like the CPPN as the genetic encoding (NEAT allows to apply the evolutionary operators)
* Geometrical patterns that define the body
  * Used in soft robots example
* Geometrical patterns that define the brain
  * AESHN 
    * the geometry, density and plasticy of an evolving neuromodulated ANN 
  * Linking problem / robot geometry with brain structure

Goal: enlighten the listeners with the CPPN's geometrical patterns
  explain how it works
  show some example patterns that it generates and the properties they have
  show how it has been used for morphologies
  show how it has been used for brains (AESHN)

## The OCRA approach

* It's all about patterns
  * Evolving the CPPN towards compliant brain / body pattern mutations
  * Simultaneous optimization of brain & body has shown to be helpful

Goal: explain how the CPPN could resolve the compliant mutations problem (RO1)

## Reflection
* Show thesis' title again and give quick summary before moving on to experiments

## Experiments

* Many carried out, here are the highlights:
* A virtual ecosystem in unity as the environment
  * Why a virtual ecosystem?
  * Current features (show dummy agent videos)
* NeuroEvolution (AESHN)
  * CartPole example > recurrent vs non recurrent brain
  * Lawnmower example > validation of the neuroevolution part
* SEMC
  * Gait learning
    * validation of the morphological (and neuroevolution) part
  * Attack
  * Sensor validation (RedWall)
    * Lower complexity after initial issues:
      * Fixed sensors to brain
      * (Crawlers: only allow morphology blocks below brain)
    * Good solution
    * Fun (and good) solution
  * In progress: Individual (simple) ecosystem survival
    * Only food
    * No water
    * Flat landscape

Goal: give an overview of the most important experiments done
  Show and discuss brain (and morph) visualizations



