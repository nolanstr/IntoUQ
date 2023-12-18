import numpy as np
import pickle
import glob
import os

from bingo.evolutionary_algorithms.generalized_crowding import \
                                                GeneralizedCrowdingEA
from bingo.selection.deterministic_crowding import DeterministicCrowding
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.stats.pareto_front import ParetoFront

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData

POP_SIZE = 1000
STACK_SIZE = 64
MAX_GEN = 10000
FIT_THRESH = -np.inf
CHECK_FREQ = 100
MIN_GEN = 500

def get_training_data():

    cwd = os.getcwd()
    tag = cwd.split("/")[-1].split("_")[-1]
    data = []
    for i in range(0, 100, 10):
        data.append(np.load(f"../../mlmc_data/mlmc_data_{i}.npy"))
    data = np.vstack(data)
    X = data[:, :-1]
    y = data[:, -1].reshape((-1,1))
    training_data = ExplicitTrainingData(x=X, y=y)
    import pdb;pdb.set_trace()    
    return training_data

def execute_generational_steps():
    
    training_data = get_training_data()

    component_generator = ComponentGenerator(training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")
    component_generator.add_operator("sqrt")
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    agraph_generator = AGraphGenerator(STACK_SIZE, 
                                       component_generator,
                                       use_pytorch=True)
    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, 
                                        algorithm='lm')

    pareto_front = ParetoFront(
                    secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    evaluator = Evaluation(local_opt_fitness, multiprocess=40)

    selection_phase = DeterministicCrowding()
    ea = GeneralizedCrowdingEA(evaluator, crossover,
                      mutation, 0.4, 0.4, selection_phase)
    
    island = Island(ea, agraph_generator, POP_SIZE, 
                                hall_of_fame=pareto_front)
    opt_result = island.evolve_until_convergence(
                                    max_generations=MAX_GEN, 
                                    fitness_threshold=FIT_THRESH, 
                                    convergence_check_frequency=CHECK_FREQ,
                                    checkpoint_base_name='checkpoint')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and \
                            ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == '__main__':
    execute_generational_steps()

