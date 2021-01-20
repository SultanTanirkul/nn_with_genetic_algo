import torch
import math
import random
import numpy as np
from numpy import genfromtxt
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pop_size = 100
num_of_bits = 15
max_num     = 2**num_of_bits
gen_number = 200
dimensions = 18
flip_prob = 1/270.0
crossover_prob = 0.5
crossover_swap_prob = 0.5
nElitists = 2
is_baldwinian = False
is_lamarckian = False
plot_pre_local_learning = False

# load the training data
train_data = genfromtxt('train.dat', delimiter=' ')
x_train = train_data[:, 0:2]
y_train = train_data[:, 2:3]
x_train = torch.as_tensor(x_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)

# load the testing data
test_data = genfromtxt('test.dat', delimiter=' ')
x_test = test_data[:, 0:2]
y_test = test_data[:, 2:3]
x_test = torch.as_tensor(x_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# set up the network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output) 

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

def generate_test_training_data_files():
    def generate_random_samples(n, a = 0, b = 2 * math.pi):
        in_samples = []
        while n > 0:
            x_1 = random.uniform(a, b)
            x_2 = random.uniform(a, b)
            in_samples.append([x_1, x_2])
            n = n - 1
        return in_samples

    def calculate_y(in_samples):
        out_samples = []
        for in_sample in in_samples:
            y = math.sin(2.0*in_sample[0] + 2.0) * math.cos(0.5 * in_sample[1]) + 0.5
            out_samples.append(y)
        return out_samples

    def combine_in_out_samples(in_samples, out_samples):
        samples = []
        for i in range(len(in_samples)):
            in_samples[i].append(out_samples[i])
            samples.append(in_samples[i])
        return samples

    def randomly_pick_samples_write_to_file(filename, data, n_samples):
        f = open(filename ,"w")
        while n_samples > 0:
            random_sample_index = random.randint(0, len(data) - 1)
            random_sample = data.pop(random_sample_index)
            line = "{} {} {}\n".format(random_sample[0], random_sample[1], random_sample[2])
            f.write(line)
            n_samples = n_samples - 1
        f.close()

    def generate_test_train_data_files(input):
        randomly_pick_samples_write_to_file("train.dat", input, 11)
        randomly_pick_samples_write_to_file("test.dat", input, 10)

    in_samples = generate_random_samples(21)
    out_samples = calculate_y(in_samples)
    samples = combine_in_out_samples(in_samples, out_samples)
    random.shuffle(samples)
    generate_test_train_data_files(samples)

def visualise_training_data():
    data = genfromtxt('train.dat', delimiter=' ')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
    ax.set_zlabel('$y$', fontsize=20)
    plt.show()

def visualise_test_data():
    data = genfromtxt('test.dat', delimiter=' ')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
    ax.set_zlabel('$y$', fontsize=20)
    plt.show()

def bin_to_real(c):
    ind_as_string = ''.join(map(str, c))
    
    num_as_int = int(ind_as_string, 2)
    num_in_range = -10 + 20 * num_as_int / max_num
    return num_in_range 

def real_to_bin(num_in_range):
    num_as_int = ((num_in_range + 10) * max_num) / 20
    num_as_int = int(round(num_as_int))
    return format(num_as_int, '015b')

def evaluate(ind_error_function_result): 
    return 1.0/(0.01 + ind_error_function_result),

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_of_bits*dimensions)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRoulette)
toolbox.register("mutate", tools.mutFlipBit, indpb=flip_prob)

def lamarckian_learning(individual, net):
    net = assign_weights_to_network(individual, net)
    net, loss_lamar = train_network(net)

    return convert_tensor_weights_to_chromosome(net)

def train_network(net):
    optimizer = torch.optim.Rprop(net.parameters(), lr = 0.01)
    loss_func = torch.nn.MSELoss(reduction='mean')
    for t in range(30):
        out = net(x_train)
        loss = loss_func(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    return net, loss.item()

def baldwinian_learning(individual, net):
    net = assign_weights_to_network(individual, net)
    net, loss_baldwinian = train_network(net)

    return loss_baldwinian

def format_weights_for_pytorch(individual):
    """Converts chromosome to weights and then formats it for pytorch"""
    # weights from input layer to hidden layer
    hidden_layer_weights = []
    for i in range(6):
        weight = bin_to_real(individual[i*15:i*15 + 15])
        hidden_layer_weights.append([weight])
    
    for i in range(6):
        weight = bin_to_real(individual[(i + 6) * 15:(i + 6) * 15 + 15])
        hidden_layer_weights[i].append(weight)
    
    # weights from hidden layer to output layer
    output_layer_weights = []
    for i in range(6):
        output_layer_weights.append(bin_to_real(individual[(i + 12) * 15:(i + 12) * 15 + 15]))

    return np.array(hidden_layer_weights, dtype=np.float32), np.array([output_layer_weights], dtype=np.float32)

def convert_tensor_weights_to_chromosome(net):
    hidden_layer_weight = net.hidden.weight.data.numpy()
    output_layer_weight = net.out.weight.data.numpy()

    chromosome = []

    for i in range(len(hidden_layer_weight[:, 0])):

        # keep number in range [-10, 10]
        if hidden_layer_weight[:, 0][i] > 10.0:
            hidden_layer_weight[:, 0][i] = 10.0
        elif hidden_layer_weight[:, 0][i] < -10.0:
            hidden_layer_weight[:, 0][i] = -10.0

        x1 = real_to_bin(hidden_layer_weight[:, 0][i])

        for i, bit in enumerate(x1):
            chromosome.append(int(bit))

    for i in range(len(hidden_layer_weight[:, 1])):

        # keep number in range [-10, 10]
        if hidden_layer_weight[:, 1][i] > 10.0:
            hidden_layer_weight[:, 1][i] = 10.0
        elif hidden_layer_weight[:, 1][i] < -10.0:
            hidden_layer_weight[:, 1][i] = -10.0

        x2 = real_to_bin(hidden_layer_weight[:, 1][i])

        for i, bit in enumerate(x2):
            chromosome.append(int(bit))
    
    for i in range(len(output_layer_weight[0])):

        # keep number in range [-10, 10]
        if output_layer_weight[0][i] > 10.0:
            output_layer_weight[0][i] = 10.0
        elif output_layer_weight[0][i] < -10.0:
            output_layer_weight[0][i] = -10.0

        v = real_to_bin(output_layer_weight[0][i])

        for i, bit in enumerate(v):
            chromosome.append(int(bit))

    return chromosome

def assign_fitness_to_individuals(individuals, net, is_initial_population = False):
    losses = get_loss_for_population(individuals, net, is_initial_population)
    fitnesses = list(map(toolbox.evaluate, losses)) # map evaluate function to each calculated loss function 

    for ind, fitness in zip(individuals, fitnesses):
        ind.fitness.values = fitness

    return individuals

def get_loss_for_population(individuals, net, is_initial_population = False):
    # a list of loss values to use for evaluation
    losses = []

    for ind in individuals:
        # No need to apply baldwian learning for initial population
        if is_baldwinian and not is_initial_population:
            loss = baldwinian_learning(ind, net)
        else:
            loss = get_loss_for_individual(ind, net, x_train, y_train)
        losses.append(loss)

    return losses

def get_loss_for_individual(individual, net, x, y):
    loss_func = torch.nn.MSELoss(reduction='mean')
    net = assign_weights_to_network(individual, net)
    
    predicted_value = net(x)
    # add MSE results to the list for further evaluation
    loss = loss_func(predicted_value, y)
    
    return loss.item()

def assign_weights_to_network(ind, net):
    # Get weights for hidden and output layers formatted for pytorch
    hidden, output = format_weights_for_pytorch(ind)
    # Assign them to the network
    net.hidden.weight = torch.nn.Parameter(torch.from_numpy(hidden))
    net.out.weight = torch.nn.Parameter(torch.from_numpy(output))

    return net

def optimise_network(pop, net):
    """Genetic Algorithm that optimises weights"""

    loss_over_generations = []
    loss_over_generations_test = []
    loss_over_generations_pre = []
    loss_over_generations_test_pre = []
    g = 0
    # Begin the optimisational process
    while g < gen_number:
        g = g + 1

        # select best individuals and apply roulette selection on the rest
        offspring = tools.selBest(pop, nElitists) + toolbox.select(pop,len(pop) - nElitists)
        offspring = list(map(toolbox.clone, offspring))

        # apply uniform crossover operator
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2, indpb=crossover_swap_prob)

        # apply mutation operator
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
        
        if plot_pre_local_learning:
            offspring = assign_fitness_to_individuals(offspring, net)
            best_network = tools.selBest(offspring, 1)[0]
            loss_over_generations_pre.append(1/best_network.fitness.values[0])
            loss = get_loss_for_individual(best_network, net, x_test, y_test)
            loss_over_generations_test_pre.append(loss)

        if is_lamarckian:
            for ind in offspring:
                lifetime_change = lamarckian_learning(ind, net)
                for i in range(len(ind)):
                    ind[i] = lifetime_change[i]

        offspring = assign_fitness_to_individuals(offspring, net)
        best_network = tools.selBest(offspring, 1)[0]
        loss_over_generations.append(1/best_network.fitness.values[0])

        # add MSE results to the list for further evaluation
        loss = get_loss_for_individual(best_network, net, x_test, y_test)

        loss_over_generations_test.append(loss)
        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return loss_over_generations, loss_over_generations_test, loss_over_generations_pre, loss_over_generations_test_pre, 


def main():
    # Generate test.dat and train.dat files
    generate_test_training_data_files()

    # 3D Visualisation
    # visualise_training_data()
    # visualise_test_data()

    net = Net(n_feature=2, n_hidden=6, n_output=1)
    net.hidden.bias = torch.nn.Parameter(torch.from_numpy(np.array([random.uniform(-10,10) for _ in range(6)], dtype=np.float32)))
    net.out.bias = torch.nn.Parameter(torch.from_numpy(np.array([random.uniform(-10,10) for _ in range(1)], dtype=np.float32)))

    pop = toolbox.population(n = pop_size)
    pop = assign_fitness_to_individuals(pop, net, True)

    # run evolutionary algorithm and then get list of MSE for the best in each generation for later plot
    loss_over_generations, loss_over_generations_test, loss_over_generations_pre, loss_over_generations_test_pre = optimise_network(pop, net)

    # Plotting graphs for MSE over generations for the best neural network for test and training samples
    plot_tain = plt.plot(np.array(loss_over_generations), 'r')
    if plot_pre_local_learning:
        plot_train_pre = plt.plot(np.array(loss_over_generations_pre), 'b')
        plt.legend(handles=(plot_tain, plot_train_pre), labels=['After Local Search', 'Before Local Search'])
    plt.xlabel("Gen")
    plt.ylabel("MSE")
    plt.title('MSE over Generations - Training')
    plt.show()

    plot_test = plt.plot(np.array(loss_over_generations_test), 'r')
    if plot_pre_local_learning:
        plot_test_pre = plt.plot(np.array(loss_over_generations_test_pre), 'b')
        plt.legend(handles=(plot_test, plot_test_pre), labels=['After Local Search', 'Before Local Search'])
    plt.title('MSE over Generations - Testing')
    plt.xlabel("Gen")
    plt.ylabel("MSE")
    plt.show()

main()
