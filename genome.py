import gym

env = gym.make("CartPole-v0")

class Genome:
    def __init__(self, net, size=4):
        self.size = size
        self.net = net
    
    def get_action_from(self, observation):
        action_array = self.net.activate(observation)
        go_to_right = int(action_array[0] > 0.5)
        return go_to_right
    
    def get_fitness(self, visualize=False):
        env.reset()
        fitness = 0.
        observation = [0]*self.size
        while True:
            if visualize:
                env.render()

            action = self.get_action_from(observation)
            observation, reward, done, _ = env.step(action)
            fitness += reward
            if done: break
    

        env.close()
        return fitness
    
    def visualize(self):
        self.get_fitness(visualize=True)