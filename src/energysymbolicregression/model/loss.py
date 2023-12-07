import numpy as np
from typing import Callable, Tuple, Union
from model.hopfield import Heigen

CleanFunctCallable = Callable[[np.ndarray, str], str]

   
class EvaluatorBase:
    def forward(self, tokenstring: str) -> Tuple[float, float]:
        # some function on tokenstring, eg eval()
        # return Tuple(y_evaluated, y_true)
        # return None to indicate failure to evaluate
        return None, 0

    def __call__(self, s):
        return self.forward(s)

class LossMetric:
    def __init__(self):
        self.metric_type = None
    def forward(self, y: float, y_o: float) -> float:
        return y - y_o
    def __call__(self, y, y_o):
        return self.forward(y, y_o)

class MSE(LossMetric):
    def __init__(self, scaler=1):
        self.metric_type = "MSE"
        self.scaler = scaler
    def forward(self, y:float, y_o:float) -> float:
        return abs(y - y_o)**2 * self.scaler

class MAE(LossMetric):
    def __init__(self, scaler=1):
        self.metric_type = "MAE"
        self.scaler = scaler
    def forward(self, y:float, y_o:float) -> float:
        return abs(y - y_o) * self.scaler


class EvalLoss:
    def __init__(self, evaluator: 'EvaluatorBase', max_str_len, num_syms, min_E=-100, metric: Union[str,'LossMetric'] = "MAE", eval_clip = 400, eval_fit_curve=1.4):
        self.max_str_len = max_str_len
        self.num_syms = num_syms
        self.nUnits = self.max_str_len*self.num_syms
        self.evaluator = evaluator
        self.min_E = min_E
        self.metric = metric
        self.eval_clip = eval_clip
        self.eval_fit_curve = eval_fit_curve

        if type(self.metric) == str:
            if self.metric == "MSE":
                self.metric = MSE()
            elif self.metric == "MAE":
                self.metric = MAE()

        self._pf = lambda s: s

        self.max_diff = 0
        self.Q = np.zeros((self.max_str_len*self.num_syms, self.max_str_len*self.num_syms)).astype('float64')


    def _set_tokenstring_preprocess_function(self, f: CleanFunctCallable):
        self._pf = f

    def _set_Q(self, Q):
        if np.mean(np.abs(Q - self.Q, casting="unsafe")) > 0.1:
            self.Q = Q
            _, V_max = Heigen.find_extreme_eigenvectors(Q)
            #V_max_binary = Heigen.closest_binary_eigenvector(V_max, self.max_str_len)
            self._set_max_diff(V_max.reshape((self.max_str_len*self.num_syms, 1)))

    def _set_max_diff(self, V_max):
            # Calculate u at equilibrium
            u_eq = np.dot(self.Q, V_max)
            
            self.max_diff = Heigen.get_diff_of_centers(u_eq)

    @staticmethod
    def scaled_log(x, d1=0, d2=1, r1=0, r2=1, c=1.4):
        """
        Scaled and shifted log function based on given parameters.

        x: Input value or array
        d1, d2: Domain range where d1 is the starting point and d2 is the ending point.
        r1, r2: Range where r1 is the starting point and r2 is the ending point.
        c: Curving constant.

        Returns the scaled and shifted sigmoid value.
        
        https://www.desmos.com/calculator/grlhshtjwt
        """

        if x - d1 < 0: #avoid complex numbers from negnum^fraction_curve_value
            return r1
        
        y = r1 + (r2 - r1) * ((x - d1) ** c) / ((d2 - d1) ** c)
        
        y = min(y, r2)

        y = max(y, r1)
        
        return y

    
    @staticmethod
    def scaled_sigmoid(x, d1=-1, d2=1, r1=-1, r2=1, c=0.125):
        """
        Scaled and shifted sigmoid function based on given parameters.

        x: Input value or array
        d1, d2: Domain range where d1 is the starting point and d2 is the ending point.
        r1, r2: Range where r1 is the starting point and r2 is the ending point.
        c: Curving constant.

        Returns the scaled and shifted sigmoid value.

        https://www.desmos.com/calculator/aytcvkhafv
        """

        sigmoid_value = 1 / (1 + np.exp((-x + (d1 + (d2 - d1) / 2)) / (c * (d2 - d1))))
        return (r2 - r1) * sigmoid_value + r1
    

    def forward(self, token_string: str, Q: np.ndarray, u: np.ndarray, V: np.ndarray, E: float) -> np.ndarray:

        self._set_Q(Q)

        #---- evaluate V to get eval loss from evaluator

        c_ts = self._pf(V, token_string)

        
        y_o, y_t = self.evaluator(c_ts)
        if y_o is None:
            y_o = self.eval_clip 

        metric_loss_value = -1 * self.metric(y_o, y_t)
        """ 
        if self.min_E is not None and E > self.min_E:
            #return 0 if energy is too high (give model time to output evaluable expressions, etc)
            return np.zeros(V.shape)

        y_o, y_t = self.evaluator(c_ts)
        if y_o is None:
            #no longer punish lack of evaluation, instead give model more time to converge using Qfuncts into evaluable output
            return np.zeros(V.shape)#-1*np.ones(V.shape)*self.eval_clip 
        metric_loss_value = -1 * self.metric(y_o, y_t)
        """
        #----- get most active neurons mask matrix

        V_reshaped = V.reshape(self.max_str_len, self.num_syms)

        row_indices = np.arange(self.max_str_len)
        col_indices = np.argmax(V_reshaped, axis=1)

        most_active_neurons_mask_reshaped = np.zeros_like(V_reshaped)
        most_active_neurons_mask_reshaped[row_indices, col_indices] = 1
        most_active_neurons_mask = most_active_neurons_mask_reshaped.reshape((self.max_str_len * self.num_syms, 1))

        
        #----- get total influence on each of the most activated neurons, to create a normalizer


        # Get the most activated values in V #in future: if making generalized hopfield ml framework, use (V > 0.5).astype(float) to not be limited by 1 per row
        #most_activated_indices = [l for l in enumerate(np.argmax(V.reshape(self.max_str_len, self.num_syms), axis=1))]

        activated_neurons_influence_sum_matrix = most_active_neurons_mask.copy()
        for pos, char_id in zip(row_indices, col_indices):
            idx = pos * self.num_syms + char_id

            sum_influence = 0

            # Check the influence of other neurons on this neuron using the Q matrix
            for other_neuron in range(Q.shape[0]):
                sum_influence += Q[other_neuron, idx]
                # using np.dot(Q, V[ids]) does not work as we don't want self connections or influence from this neuron to other neurons, only others to this (in case non-mutual connections)

            activated_neurons_influence_sum_matrix[pos * self.num_syms + char_id] *= sum_influence * -1

        max_sum_abs_influence = np.max(np.abs(activated_neurons_influence_sum_matrix[np.where(np.abs(activated_neurons_influence_sum_matrix) > 0)]))


        #print("activated_neurons_influence_sum_matrix: ")
        #print(activated_neurons_influence_sum_matrix.reshape((self.max_str_len, self.num_syms)))


        #----- create spread_neurons_loss_matrix with normalizing scalars such that sum_influence for
        #      each activated_neurons_influence_sum_matrix would be max_sum_abs_influence. this matrix
        #      will hold the amount of influence the neuron at that position has on the most active neurons

        spread_neurons_loss_matrix = np.zeros_like(u)

        #print(f"max_sum_abs_influence: {max_sum_abs_influence}")

        for pos, char_id in zip(row_indices, col_indices):
            idx = pos * self.num_syms + char_id
            scaler = abs(activated_neurons_influence_sum_matrix[pos * self.num_syms + char_id]) / max_sum_abs_influence
            #print(f"pos 1 scaler: {scaler}")
            for other_neuron in range(Q.shape[0]):
                spread_neurons_loss_matrix[other_neuron] += Q[other_neuron, idx] * scaler
                #if abs(Q[other_neuron, idx]) > 0:
                    #print(f"snlm at {(other_neuron//self.num_syms, other_neuron%self.num_syms)} was appended by {Q[other_neuron, idx]} * {scaler} = {Q[other_neuron, idx] * scaler}. is now {spread_neurons_loss_matrix[other_neuron]}")


        #----- set all most activated neurons to be max_sum_abs_influence

        activated_neurons_loss_matrix = most_active_neurons_mask * max_sum_abs_influence

        #print("activated_neurons_loss_matrix")
        #print(activated_neurons_loss_matrix.reshape((self.max_str_len, self.num_syms)))

        #print("spread_neurons_loss_matrix")
        #print(spread_neurons_loss_matrix.reshape((self.max_str_len, self.num_syms)))

        #----- get full loss matrix

        unsquashed_loss_matrix = activated_neurons_loss_matrix + spread_neurons_loss_matrix
        unsquashed_loss_matrix *= -1

        loss_matrix_base = self.scaled_sigmoid(unsquashed_loss_matrix, d1=-1*max_sum_abs_influence, d2=max_sum_abs_influence, r1=-1, r2=1, c=0.125)

        #print("unsquashed loss matrix: ")
        #print(unsquashed_loss_matrix.reshape((self.max_str_len, self.num_syms)))
        
        #----- scale eval loss based on Q density, current convergence state, and influence of Q@V on normal model learning

        

        #use diff value of activated u vs inactive u to determine model convergence state (greater value = more converged = loss should be more influential)
        diff = Heigen.get_diff_of_centers(u)

        # normalize between 0 and 1 using magic. 
        # This is essentially a measure of convergence. When 0, model is not converged, when 1, model is very converged
        convergence = diff / self.max_diff

        # max domain of input specified by user based on evaluator 
        ld2 = self.eval_clip

        # abs(qv_nonzero_mean) represents middle point of proposed model changes by Q matrix; scale max y to be at most this
        lr2 = convergence # * abs(qv_nonzero_mean)# 10*abs(qv_nonzero_mean) + 3 #min((qv_min)*-1, 0) + 3

        # squash loss value between 0 and current convergence percentage
        y_s = -1*self.scaled_log(-1*metric_loss_value, d1=0, d2=ld2, r1=0, r2=lr2, c=self.eval_fit_curve) #r1=qv_min-1, r2=qv_max+1, c=self.eval_fit_curve)


        #print("squashed loss matrix:")
        #print(loss_matrix_base.reshape((self.max_str_len, self.num_syms)))


        #scale loss by internal energy; when model over threshold of convergence, loss (external energy) will be more influential than internal energy (q@v)
        #Q@V scaler; scale depending on values in Q@V
        # * 2 = potentially twice as influential as Q@V
        qv = Q@V
        internal_energy_scaler = np.mean(np.abs(qv)) * 2
        
        #full loss matrix
        loss_matrix = loss_matrix_base * y_s * internal_energy_scaler * -1

        #print("eval_applied loss matrix:")
        #print(loss_matrix.reshape((self.max_str_len, self.num_syms)))


        print(f"loss: {metric_loss_value}, udiff: {diff}, max_udiff: {self.max_diff}, convergence: {convergence}, internal energy scaler: {internal_energy_scaler}")

        return loss_matrix


    def __call__(self, token_string: str, Q: np.ndarray, u: np.ndarray, V: np.ndarray, E: float) -> np.ndarray:
        return self.forward(token_string, Q, u, V, E)

