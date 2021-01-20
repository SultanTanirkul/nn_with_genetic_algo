import random
import numpy
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from deap import creator, base, tools, algorithms
from itertools import chain
from operator import attrgetter, itemgetter
from deap.benchmarks.tools import diversity, convergence, hypervolume
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

num_of_gen = 30
num_of_bits = 10
pop_size    = 28
dimensions  = 3
max_num     = 2**num_of_bits
flipProb    = 1 / (num_of_bits * dimensions)
crossover_prob = 0.9

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_of_bits*dimensions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)

def efficient_nd_sort(individuals, n, first_front_only = False):
    # return empty array if there are no individuals provided
    if len(individuals) == 0:
        return []

    # create list to hold pareto fronts
    pareto_fronts = []

    # sort individuals by f_1
    individuals.sort(key=lambda x:x.fitness.values[0])

    # append first solution to the resulting poretto front
    pareto_fronts.append([individuals[0]])
    
    # start sorting from second individual
    for individual in individuals[1:]:
        k = efficient_nd_sort_single_iteration(individual, pareto_fronts)
        
        # create new pareto front if all individuals in the existing pareto_fronts dominate current individual
        if k + 1 > len(pareto_fronts):
            pareto_fronts.append([individual])
        # otherwise add to the k's pareto_front
        else:
            pareto_fronts[k].append(individual)

    # Keep only the fronts required to have n individuals.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= n:
                return pareto_fronts[:i+1]
        return pareto_fronts
    else:
        return pareto_fronts[0]

def efficient_nd_sort_single_iteration(individual, pareto_fronts):
    x = len(pareto_fronts)
    k = 0

    while True:
        for pareto_front in pareto_fronts:
            #compare individual with the given poretto front
            if front_does_not_dominate_ind(pareto_front, individual):
                return k
            else:
                k = k + 1
                if k + 1 > x:
                    return x

def front_does_not_dominate_ind(pareto_front, ind):
    for i in reversed(range(len(pareto_front))):
        if pareto_front[i].fitness.dominates(ind.fitness):
            return False
    return True

def calc_fitness(individual):
    # Separate decision variables
    x1 = individual[0:10]
    x2 = individual[10:20]
    x3 = individual[20:30]
    
    # Decode decision variables
    x1 = chrom_to_real(x1)
    x2 = chrom_to_real(x2)
    x3 = chrom_to_real(x3)
    
    # Calculate fitness
    f1 = ((x1/2.0)**2+(x2/4.0)**2+x3**2)/3.0
    f2 = ((x1/2.0-1.0)**2+(x2/4.0-1.0)**2+(x3-1.0)**2)/3.0    

    return f1, f2

def modSelNSGA2(individuals, k, nd='standard'):
    """Modified version of selNSGA2 algorithm from DEAP library that
    supports efficient non-dominated sorting algorithm implemented in
    this file. The standard algorithm was changed from fast to efficient nd.
    """
    if nd == 'standard':
        pareto_fronts = efficient_nd_sort(individuals, k)
    elif nd == 'fast':
        pareto_fronts = tools.sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = tools.sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen

def assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist

toolbox.register("evaluate", calc_fitness)
toolbox.register("select", modSelNSGA2)

def chrom_to_real(c):
    ind_as_string = ''.join(map(str, c))
    degray = gray_to_bin(ind_as_string)
    num_as_int = int(degray, 2) # convert to int from base 2 list
    num_in_range = -4 + 8 * num_as_int / max_num
    return num_in_range 

def get_worst_objectives(individuals):
    worst_f1 = individuals[0].fitness.values[0]
    worst_f2 = individuals[0].fitness.values[1]
    for ind in individuals[1:]:
        if worst_f1 < ind.fitness.values[0]:
            worst_f1 = ind.fitness.values[0]
        if worst_f2 < ind.fitness.values[1]:
            worst_f2 = ind.fitness.values[1]
    return [worst_f1, worst_f2]

hyper_volume = []

def main():
    random.seed(42)
    pop = toolbox.population(n = pop_size)
    fitnesses = toolbox.map(toolbox.evaluate, pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    worst_objectives = get_worst_objectives(pop)

    # Begin the generational process
    for gen in range(num_of_gen):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))

        # selTournamentDCD means Tournament selection based on dominance (D) 
        # followed by crowding distance (CD). This selection requires the 
        # individuals to have a crowding_dist attribute
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            #make pairs of all (even,odd) in offspring
            toolbox.mate(ind1, ind2, crossover_prob)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
        
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit        

        front_parent = numpy.array([ind.fitness.values for ind in pop])
        front_offspring = numpy.array([ind.fitness.values for ind in offspring])
        
        # show plots only for the first generation
        if gen == 1:
            plt.title("Parent and Offspring Population")
            scatter_parent = plt.scatter(front_parent[:,0], front_parent[:,1], c="b", alpha=0.5)
            scatter_offspring = plt.scatter(front_offspring[:,0], front_offspring[:,1], c="r", alpha=0.5)
            plt.legend(handles=(scatter_parent, scatter_offspring), labels=['Parent', 'Offspring'])
            plt.xlabel("f_1")
            plt.ylabel("f_2")
            plt.show()

        # Select the next generation population
        pop = toolbox.select(pop + offspring, pop_size)

        # show plots only for the first generation
        if gen == 1:
            selected_front = numpy.array([ind.fitness.values for ind in pop])
            plt.scatter(front_parent[:,0], front_parent[:,1], c="b", alpha=0.5)
            scatter_parent = plt.scatter(front_offspring[:,0], front_offspring[:,1], c="b", alpha=0.5)
            scatter_selected = plt.scatter(selected_front[:,0], selected_front[:,1], c="r", alpha=0.5)
            plt.legend(handles=(scatter_parent, scatter_selected), labels=['Parent+Offspring', 'Selected'])
            plt.title("Parent + Offspring and Selected Population")
            plt.xlabel("f_1")
            plt.ylabel("f_2")
            plt.show()
        
        hyper_volume.append(hypervolume(pop, worst_objectives))
    return pop, hyper_volume



if __name__ == "__main__":
    pop, hyper_volume = main()

    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.plot(range(num_of_gen), hyper_volume)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel("Generations")
    plt.ylabel("Hypervolume")
    plt.show()
    pop.sort(key=lambda x: x.fitness.values)

    front = numpy.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.title("The Final Objective Space")
    plt.xlabel("f_1")
    plt.ylabel("f_2")
    plt.axis("tight")
    plt.show()