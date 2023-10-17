# EnergySymbolicRegression

This research project explores the integration of symbolic regression into energy-based models (EBMs) using Hopfield networks. 

Traditional symbolic regression algorithms are used largely for modeling mathematical functions from data. Energy-Based Symbolic Regression, however, allows for more flexibility by allowing the evaluation loss of the generated string to directly backprop into the model, while also not being limited to being trained from data sources. Just as well, through hebbian learning, this  model architecture can quickly converge on evaluable solutions, compared to other traditional methods of symbolic regression.

The proposed methodology hinges on "Qfunctions," which are used to determine the behavior of the connectivity matrix and subsequently influencing the system's interpretation process. These Qfunctions will be generalized into a language, allowing users to program the space of the output string's syntax. Users will also be allowed to code their own evaluation methods based on implementation of the model.

By leveraging the energy dynamics inherent to Hopfield networks, the model identifies configurations corresponding to minimal energy states, representing optimal symbolic approximations of the "syntax space". Just as well, specialized loss analytics are designed and implemented to allow the evaluation of generated strings to interact with traditional hebbian learning formulas, allowing the model to approach specific solutions. 

A significant aspect of this project is the development of a programming language specifically designed for these EBMs. This language aims to introduce fine control over the syntax of output token strings, offering a nuanced approach to symbolic regression tasks. 

Todo:
- Finish optimizer
- Abstract above QFunctions for more generalizable programming language
- Generalize the model for multidimensional problems (beyond just 2), and for extending beyond symbolic regression while maintaining the QFunctions language, losses, and optimizers
- Solve Q^n problem, allowing pairs of activations to influence other neurons 

Check out the notebook: https://colab.research.google.com/drive/1ctDR0GwX0pwPU11Fwp6ma1BhfYo6VS5a?usp=sharing

