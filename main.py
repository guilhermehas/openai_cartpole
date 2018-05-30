from __future__ import print_function
import os
import neat
import visualize
import gym

env = gym.make("CartPole-v0")

def get_action_from(observation, net):
    action_array = net.activate(observation)
    go_to_right = int(action_array[0] > 0.5)
    return go_to_right

def get_fitness(net):
    fitness = 0.
    env.reset()
    observation = [0,0,0,0]
    while True:
        observation, reward, done, _ = env.step(get_action_from(observation, net=net))
        fitness += reward
        if done: break
    return fitness

def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = 0.
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = get_fitness(net)


def visualize_in_action(net):
    env.reset()
    observation = [0,0,0,0]
    for _ in range(10**4):
        env.render()
        
        action = get_action_from(observation, net)
        observation, _, done, _ = env.step(action)

        if done: break
    
    env.close()


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    print(f'\nFitness winner: {get_fitness(winner_net)}')
    visualize_in_action(winner_net)
    

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
