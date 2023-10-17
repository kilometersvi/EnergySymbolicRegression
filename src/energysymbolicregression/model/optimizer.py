import numpy as np

class Optimizer:
    """
    optimizer. treats L like I in u * Q @ V + I. Loss is found for small changes in V and accumulated.
    """
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def get_partials(self, L: np.ndarray, V: np.ndarray, dt: float = 0.1) -> np.ndarray:

        dV = np.zeros_like(V)
        for i in range(V.shape[0]):

            # Calculate partial derivative
            V[i] += dt
            L_new = self.model.get_evalloss(V=V)
            V[i] -= dt

            try:
                partial_derivative = (L_new - L) / dt
            except Warning as w:
                print(f'{w}, {dt}, {partial_derivative}')

            # Update dV
            dV[i] = partial_derivative[i]

        return dV

    def process(self, L: np.ndarray, V: np.ndarray) -> np.ndarray:

        dV = self.get_partials(L, V)
        L = L.copy()

        L += dV

        return L

class AdaptivePartialsOptimizer(Optimizer):
    """
    modified partial derivative function which will determine an ideal dT at each neuron,
    and only compute loss for that neuron if we know an increased activation to that neuron
    will change the loss (otherwise we know loss won't change)
    """
    def __init__(self, std_width: float = 2):
        #std_width = how many standard deviations to include for increasing activation
        #model only needed for self.max_str_len, self.num_syms, self.chars
        super().__init__()
        self.std_width = std_width

    def get_partials(self, L: np.ndarray, V: np.ndarray, dt: float = 0.1) -> np.ndarray:
        # skip calculating partials that won't change output.

        dV = np.zeros_like(V)
        for i in range(self.model.max_str_len):
            activations_row = V[i*self.model.num_syms:(i+1)*self.model.num_syms]
            pos_max, n_max_row = np.argmax(activations_row), np.max(activations_row)

            #find appropriate activation adjustment, such that very inactive neurons stay inactive, but competing neurons may be activated
            mean_activation = np.mean(activations_row)
            std_activation = np.std(activations_row)
            eps = mean_activation + self.std_width * std_activation + 0.01

            for j, c_char in enumerate(self.model.chars):
                c_n = V[i*self.model.num_syms + j]

                if c_n + eps >= n_max_row and j != pos_max:  # only proceed if epsilon is large enough to potentially change the max neuron, else skip to save compute
                    # Calculate partial derivative

                    V[i*self.model.max_str_len + j] += eps
                    L_new = self.model.get_evalloss(V=V)
                    V[i*self.model.max_str_len + j] -= eps

                    try:
                        partial_derivative = (L_new - L) / eps
                    except Warning as w:
                        print(f'{w}, {eps}, {partial_derivative}')

                    # Update dV
                    dV[i*self.model.max_str_len + j] = partial_derivative[i*self.model.max_str_len + j]

        return dV



class MaskedLossOptimizer(AdaptivePartialsOptimizer):
    """
    modified process for use with masked loss matrices.
    in masked loss matrix, all values are 0 except neurons that are activated (the most activated
    neurons of each position). as such, in the case a change in V is found to decrease loss,
    setting 0 to some value that is less than original loss, but still greater than 0, indicates
    to the model that this increase in activation should still be inhibited. Not ideal. Instead,
    we set the loss at this value to the dV, such that dV[i] > 0 > loss[already activated neuron].
    kinda hacky and nonmathematical but whatevs.
    """
    def __init__(self, std_width: float = 2, decrease_loss_scaler: float = 0.3):
        super().__init__(std_width=std_width)
        self.decrease_loss_scaler = decrease_loss_scaler

    def process(self, L: np.ndarray, V: np.ndarray) -> np.ndarray:
        #partial derivatives of loss
        dV = self.get_partials(L, V)
        L = L.copy()

        #for masked loss
        if (np.abs(dV).any() > 0.01): print('dV was found...')

        for i in range(self.model.max_str_len):
            for j, c in enumerate(self.model.chars):
                if dV[i*self.model.max_str_len + j] < 0:
                    L[i*self.model.max_str_len + j] = -1 * dV[i*self.model.max_str_len + j] * self.decrease_loss_scaler
                elif dV[i*self.model.max_str_len + j] > 0:
                    L[i*self.model.max_str_len + j] += dV[i*self.model.max_str_len + j]

        return L


