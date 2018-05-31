# Open AI - CartPole - v0
This project is a resolution of the problem of [Open AI CartPole](https://gym.openai.com/envs/CartPole-v0/) using Neuro Evolution.
It is a combination of genetic algorithm with neural network.

## Prerequisites
To run the project, you can install python 3 and it's libraries or you can run with Docker.

### Running with python 3
```sh
python3 main.py
```

### Running with Docker
```sh
docker run guilhermehas/openai_cartpole
```

## Examples

### Example of population
```
Population's average fitness: 19.12667 stdev: 28.66457
Best fitness: 200.00000 - size: (1, 4) - species 1 - id 42

Best individual in generation 0 meets fitness threshold - complexity: (1, 4)

Best genome:
Key: 42
Fitness: 200.0
Nodes:
        0 DefaultNodeGene(key=0, bias=0.30791014303939324, response=1.0, activation=sigmoid, aggregation=sum)
Connections:
        DefaultConnectionGene(key=(-4, 0), weight=1.9398153633922424, enabled=True)
        DefaultConnectionGene(key=(-3, 0), weight=2.6618273702667237, enabled=True)
        DefaultConnectionGene(key=(-2, 0), weight=1.3327123904146594, enabled=True)
        DefaultConnectionGene(key=(-1, 0), weight=0.3936091043666671, enabled=True)

Output:

Fitness winner: 200.0
```