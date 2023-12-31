# Symbolic Regression with Hebbian Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kilometersvi/EnergySymbolicRegression/blob/main/src/energysymbolicregression/notebooks/math.ipynb)

This research project explores the integration of symbolic regression into energy-based models (EBMs) using Hopfield networks. 

Traditional symbolic regression algorithms are used largely for modeling mathematical functions from data. Energy-Based Symbolic Regression, however, allows for more flexibility by allowing the evaluation loss of the generated string to directly backprop into the model, while also not being limited to being trained from data sources. Just as well, through hebbian learning, this  model architecture can quickly converge on evaluable solutions, compared to other traditional methods of symbolic regression.

The proposed methodology hinges on "Qfunctions," which are used to determine the behavior of the connectivity matrix and subsequently influencing the system's interpretation process. These Qfunctions will be generalized into a language, allowing users to program the space of the output string's syntax. Users will also be allowed to code their own evaluation methods based on implementation of the model.

By leveraging the energy dynamics inherent to Hopfield networks, the model identifies configurations corresponding to minimal energy states, representing optimal symbolic approximations of the "syntax space". Just as well, specialized loss analytics are designed and implemented to allow the evaluation of generated strings to interact with traditional hebbian learning formulas, allowing the model to approach specific solutions. 

A significant aspect of this project is the development of a programming language specifically designed for these EBMs. This language aims to introduce fine control over the syntax of output token strings, offering a nuanced approach to symbolic regression tasks. 

## How it works

At the heart of our model is the classic Hopfield network update rule. However, wae've introduced a significant change to incorporate the evaluation loss, allowing the model to prioritize solution convergence based on this loss.

The original Hopfield update formula is given by:

$$\frac{du_i}{dt} = -u_i + \sum^n_{j=0}V_j Q_{ij} + I_i $$

In this equation:
- $\frac{du}{dt}$ represents the change in activation over time (in our case, over this current epoch)
- $u_i$ is the current activation of neuron k.
- $V$ is the vector of current neuron activations.
- $Q$ is the weight matrix representing the interactions between neurons.
- $I$ is the bias term, traditionally a constant for all neurons.

Our modification involves replacing $I$ with a loss matrix, which we compute based on how well the current state of the network solves the desired task. This loss matrix takes on the role of the bias, where higher loss values indicate stronger biases for certain neurons. Essentially, the loss steers the network towards states that better satisfy our goals.

By treating the output loss as the bias for the current epoch of solution convergence, we can guide the network's dynamics towards producing more accurate and relevant solutions. 

Learn more about hopfield networks: 
- https://www.youtube.com/watch?v=HoWJzeAT9uc
- https://arxiv.org/abs/2008.02217
- http://www.scholarpedia.org/article/Hopfield_network

Todo:
- Finish optimizer
- Abstract above QFunctions for more generalizable programming language
- Generalize the model for multidimensional problems (beyond just 2), and for extending beyond symbolic regression while maintaining the QFunctions language, losses, and optimizers
- Solve $Q^n$ problem, allowing pairs of activations to influence other neurons
- Create automatic learner for Q function. Explore opportunities for understanding maximum adjustment for each weight class without disrupting model convergence (use minimum eigenvalue's eigenvector, or re-derive based on original Hopfield equation for maximum number of memories stored in Q matrix)
- Create platform for benchmarking
- After these are done, transition to Modern Hopfield Model over Graded Hopfield Model like current implementation. Not useful for symbolic regression, but will allow this model to serve as a machine learning layer after being refactored into PyTorch, allowing us to explore benefits of this layer for explainable machine learning.

Overview:
https://youtu.be/ZN9ygXVR9es?si=EKaEQP-YdozNyzUG

Technical:
https://youtu.be/fQ5_cFzkUm4?si=1E0cAzed02X6TjUF

Presentation:
https://docs.google.com/presentation/d/1bTF0vkGg6Gqe22dvT-G8K5KXc0e9I0C9VgxUPeJxMd4/edit?usp=sharing
