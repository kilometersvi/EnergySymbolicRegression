import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict
from typing import Callable, List, Dict, Union
from matplotlib.animation import FuncAnimation
from model.factories import QFactory, IFactory
from model.optimizer import *
from model.loss import *
from model.onehotencode import OneHotEncoder
from model.hopfield_utils import *




QFunctCallable = Callable[['QFactory', int, List[str]], None]
IFunctCallable = Callable[['IFactory', int, List[str]], None]

class H_SymReg:
    def __init__(self, chars: List[str], max_str_len: int = 10, gain: float = 999, dt: float = 0.01, min_energy_for_eval: float = -100, eval_clip: float = 400,
                 Qfuncts: List[QFunctCallable] = None, Ifuncts: List[IFunctCallable] = None,
                 loss: 'EvalLoss' = None, optimizer: 'Optimizer' = None, evaluator: 'EvaluatorBase' = None,
                 conf: Dict[str, float] = None, sets: Dict[str, List[str]] = None,
                 cleanfunct: CleanFunctCallable = None):

        if not chars:
            raise AttributeError("chars must be defined")

        self.encoder = OneHotEncoder(chars)
        self.chars = chars
        self.max_str_len = max_str_len
        self.num_syms = len(chars)
        self.nUnits = max_str_len * self.num_syms
        self.gain = gain
        self.dt = dt
        self.u = (np.random.rand(self.nUnits, 1) - 0.5) * 0.01 + 0.5
        self.V = self.squasher(self.u, self.gain)
        self.E_hist = [np.inf]
        self.V_hist = [self.V]
        self.u_hist = [self.u]
        self.L_hist = [np.zeros_like(self.V)]

        #I & Q
        self.conf = conf
        self.sets = sets

        self.ifactory = IFactory(self.max_str_len, self.chars, self.conf, self.sets)
        if Ifuncts is None or len(Ifuncts) == 0:
            self.Ifuncts = []
        else:
            self.Ifuncts = Ifuncts
            for ifunc in self.Ifuncts:
                ifunc(self.ifactory, self.max_str_len, self.chars)
        self.I = self.ifactory.I

        self.qfactory = QFactory(self.max_str_len, self.chars, self.conf, self.sets)
        if Qfuncts is None or len(Qfuncts) == 0:
            self.Qfuncts = []
            raise Warning("Qfuncts not defined- do you really want to train on empty connectivity matrix?")
        else:
            self.Qfuncts = Qfuncts
            for qfunc in self.Qfuncts:
                qfunc(self.qfactory, self.max_str_len, self.chars)
        self.Q = self.qfactory.Q


        #losses & optimizer
        if evaluator is None:
            raise AttributeError("Evaluator is not set.")
        self.evaluator = evaluator

        if cleanfunct is None:
            cleanfunct = lambda s: s
        self.clean_output = cleanfunct

        if loss is None:
            self.get_loss = EvalLoss(self.evaluator, self.max_str_len, self.num_syms, eval_clip=eval_clip, min_E=min_energy_for_eval)
        else:
            self.get_loss = loss
        self.get_loss._set_tokenstring_preprocess_function(self.clean_output)


        self.optimizer = optimizer
        if optimizer is not None:
            self.optimizer.set_model(self)

        self._set_internal_energy_domain()


    def _set_internal_energy_domain(self):

        V_max, V_min = find_extreme_eigenvectors(self.Q)

        # k = number of active neurons; in this implementation of hopfield, we already know what this is: the number of output positions
        k = self.max_str_len

        closest_possible_V_max = closest_binary_eigenvector(V_max, k)
        closest_possible_V_min = closest_binary_eigenvector(V_min, k)

        self.V_extremes = (closest_possible_V_min.reshape((self.max_str_len*self.num_syms, 1)), 
                           closest_possible_V_max.reshape((self.max_str_len*self.num_syms, 1)))

        print(f"(max&min) V shape: {self.V_extremes[0].shape}, Q shape: {self.Q.shape}")

        max_E = calc_internal_energy(self.Q, self.V_extremes[1])
        min_E = calc_internal_energy(self.Q, self.V_extremes[0])

        self.energy_domain = (min_E, max_E)

        print(self.energy_domain)

        self.get_loss._set_max_diff(self.V_extremes[1])


    @staticmethod
    def squasher(u, gain):
        return 0.5 * (1 + np.tanh((u - 0.5) * gain))

    def get_evalloss(self, eq: str = None, Q: np.ndarray = None, u: np.ndarray = None, V: np.ndarray = None, E: float = None):
        if Q is None: Q = self.Q
        if u is None: u = self.u
        if V is None: V = self.V
        if eq is None: eq = self.decode_output(V=V, clean=True)
        if E is None: E = self.E_hist[-1]

        return self.get_loss.forward(eq, Q, u, V, E)


    def evaluate(self, eq: Union[str, np.ndarray]):
        if type(eq) == str:
            pass
        elif type(eq) == np.ndarray:
          eq = self.decode_output(V=eq)

        else:
            raise TypeError("Invalid arguement type in position 2")

        ys = self.evaluator(eq)
        if ys is None:
            return None
        else:
            return ys[0]


    def update(self, I=None, n_iters=1000, min_dE=0.0001, earlystopping=50, min_E=-130):
        """
        I = bias
        n_iters = epochs
        min_dl = minimum change in L before noticing change in L
        earlystopping = number of iters without change before quitting
        min_L = minimum energy before quitting
        """
        if I is None:
            I = self.I

        min_found_E = np.inf
        lastBigJump = 0


        for e in range(n_iters):

            global c_iter
            c_iter = e

            # evallosses
            L = self.get_evalloss(eq = self.decode_output(V=self.V, clean=True))

            if self.optimizer is not None:
                L = self.optimizer.process(L, self. V)

            self.L_hist.append(L)

            du = -self.u + self.Q @ self.V + I + L
            self.u += du * self.dt
            self.V = self.squasher(self.u, self.gain)

            # energy calc
            e1 = -0.5 * np.dot(self.V.T, np.dot(self.Q, self.V))
            e2 = np.dot(self.V.T, I+L)

            print(f"internal energy: {e1}, external energy: {e2}")
            if e == 0:
                print(f"V shape: {self.V.shape}, Q shape: {self.Q.shape}, I shape: {I.shape}, L shape: {L.shape}")
            E = (-0.5 * np.dot(self.V.T, np.dot(self.Q, self.V)) - np.dot(self.V.T, I+L))[0][0]
            #print(f"{i},{np.dot(self.V.T, new_I)}")

            self.E_hist.append(E)
            self.u_hist.append(self.u.copy())
            self.V_hist.append(self.V)

            if len(self.E_hist) > 1:
                dE = self.E_hist[-1] - self.E_hist[-2]

                if np.abs(dE) < min_dE:
                    lastBigJump = e
                min_found_E = min(min_found_E, E)


                #print(f"{(np.abs(dL) <= min_dL)}, {(L < minL)}, {(i - lastBigJump) >= 100}")
                #if ((np.abs(dL) <= min_dL) and (L < minL)) or ((i - lastBigJump) >= 100):
                #    break
                if (e - lastBigJump) >= earlystopping:
                    break
                if min_E is not None and E < min_E:
                    break

                if e%(n_iters//100)==0:
                    print(f'{e}: {self.decode_output(self.V, clean=False)}')


    def plot_results(self):
        V_reshaped = self.V.reshape(self.max_str_len, self.num_syms)
        plt.imshow(V_reshaped)
        plt.show()


    def plot_Ehist(self, max_y=None):
        #plt.set_xlim((1,len(self.L_hist)))
        plt.plot(np.arange(1, len(self.E_hist)), self.E_hist[1:])
        if max_y is not None:
            plt.ylim((np.min(self.E_hist[1:])-5, min(np.max(self.E_hist[1:])+5,max_y)))#np.max(self.E_hist[1:])+5))

        plt.show()


    def plot_histories_as_video(self):

        fig, axes = plt.subplots(2, 3, figsize=(8, 10))

        axes[0, 0].set_title('V')
        axes[0, 1].set_title('u')
        axes[1, 1].set_title('Eval Loss Matrix')
        axes[1, 0].set_title('Q@V Matrix')
        axes[0, 2].set_title('Energy')


        #generate Q@V
        QV_hist = [self.Q@v for v in self.V_hist]

        # Reshape grid datas
        c_V_hist = [v.reshape(self.max_str_len, self.num_syms) for v in self.V_hist]
        c_u_hist = [u.reshape(self.max_str_len, self.num_syms) for u in self.u_hist]
        c_L_hist = [l.reshape((self.max_str_len, self.num_syms)) for l in self.L_hist]
        c_QV_hist = [qv.reshape((self.max_str_len, self.num_syms)) for qv in QV_hist]

        # Get global min and max for consistent color limits
        global_min_V = min(matrix.min() for matrix in c_V_hist)
        global_max_V = max(matrix.max() for matrix in c_V_hist)
        global_min_u = min(matrix.min() for matrix in c_u_hist)
        global_max_u = max(matrix.max() for matrix in c_u_hist)
        global_min_L = min(matrix.min() for matrix in c_L_hist)
        global_max_L = max(matrix.max() for matrix in c_L_hist)
        global_min_QV = min(matrix.min() for matrix in c_QV_hist[5:])
        global_max_QV = max(matrix.max() for matrix in c_QV_hist[5:])

        im_V = axes[0, 0].imshow(c_V_hist[0], cmap='viridis', vmin=global_min_V, vmax=global_max_V)
        im_u = axes[0, 1].imshow(c_u_hist[0], cmap='viridis', vmin=global_min_u, vmax=global_max_u)
        im_L = axes[1, 1].imshow(c_L_hist[0], cmap='viridis', vmin=global_min_L, vmax=global_max_L)
        im_QV = axes[1, 0].imshow(c_QV_hist[0], cmap='viridis', vmin=global_min_QV, vmax=global_max_QV)
        line, = axes[0, 2].plot([], [])

        #fig.colorbar(im_V, ax=axes[0, 0])
        fig.colorbar(im_u, ax=axes[0, 1])
        fig.colorbar(im_L, ax=axes[1, 1])
        fig.colorbar(im_QV, ax=axes[1, 0])

        def animate(i):
            s = self.decode_output(V=c_V_hist[i].reshape((self.nUnits,1)))
            y = self.evaluate(s)
            if y is None: y = "err"
            s += f" = {y}"
            fig.suptitle(f"{s}")
            
            im_V.set_array(c_V_hist[i])
            im_u.set_array(c_u_hist[i])
            im_L.set_array(c_L_hist[i])
            im_QV.set_array(c_QV_hist[i])
            
            if i >= 1:
                line.set_data(np.arange(1, i + 1), self.E_hist[1: i + 1])
                axes[0, 2].relim()
                axes[0, 2].autoscale_view(True, True, True)


        anim = FuncAnimation(fig, animate, frames=len(c_V_hist), interval=40, repeat=False)

        from IPython.display import HTML
        v = anim.to_html5_video()

        try:
            # saving to m4 using ffmpeg writer 
            writervideo = anim.FFMpegWriter(fps=60) 
            anim.save('outvid.mp4', writer=writervideo) 
            plt.close() 
        except Exception as e:
            print(f"error saving video: {e}")
        return HTML(v)


    def decode_output(self, V: np.ndarray = None, clean: bool = True):
        if V is None:
            V = self.V
        reshaped_V = V.reshape(self.max_str_len, self.num_syms)
        o =  self.encoder.decode(reshaped_V)

        if clean:
            o = self.clean_output(V=V, expr=o)

        return o
