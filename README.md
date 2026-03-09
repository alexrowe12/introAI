# Intro to AI

A collection of projects from my Introduction to Artificial Intelligence course, covering multi-armed bandits, Markov decision processes, and neural network classifiers.

## Projects

### BanditProject

Implementation of the **multi-armed bandit problem** using Thompson Sampling with Dirichlet priors. Features social learning from other agents' choices and Monte Carlo simulation methods.

**Key files:**
- `bandit.py` - Thompson Sampling with Dirichlet priors
- `BayesUCB.py` - Bayesian Upper Confidence Bound implementation
- `monte_carlo.py` - Monte Carlo simulation methods
- `simulator.py` - Bandit simulation environment

### BucketsProject

A **Markov Decision Process (MDP)** solution to the classic water bucket problem. Given three buckets of different capacities, the goal is to find the optimal policy for reaching a target water level through fill, empty, and pour actions.

**Key files:**
- `buckets_mdp.py` - MDP implementation with value iteration

### PerceptronProject

Image classification using three progressively complex neural network architectures:

1. **Single Perceptron** - Basic linear classifier
2. **Multi-Layer Perceptron (MLP)** - Feedforward neural network
3. **Convolutional Neural Network (CNN)** - Deep learning approach

Supports GPU acceleration on M1 Macs via Metal.

**Key files:**
- `classifier1_perceptron.py` - Single perceptron classifier
- `classifier2_mlp.py` - MLP classifier
- `classifier3_cnn.py` - CNN classifier

**Usage:**
```bash
# Train a model
python classifier2_mlp.py --train --epochs 50 --gpu

# Classify images
python classifier2_mlp.py -d [image_directory]
```
