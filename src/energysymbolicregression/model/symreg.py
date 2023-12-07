# Import necessary classes from Hopfield.py
import numpy as np
from model.hopfield import GHN, Heigen
from model.onehotencode import OneHotEncoder
from model.loss import EvalLoss, EvaluatorBase
from model.factories import QFactory, IFactory
from model.optimizer import *
from typing import List, Dict, Callable, Union
from easytools.decorators import flexmethod
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt



QFunctCallable = Callable[['QFactory', int, List[str]], None]
IFunctCallable = Callable[['IFactory', int, List[str]], None]
CleanFunctCallable = Callable[['np.ndarray', str], str]

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

        # Initialize GHN instance
        self.ghn = GHN(Q=None, u_shape=(self.max_str_len, self.num_syms), gain=gain, dt=dt)

        self.E_hist = [np.inf]
        self.conf = conf
        self.sets = sets

        # Initialize I & Q factories
        self.ifactory = IFactory(self.max_str_len, self.chars, self.conf, self.sets)
        self.Ifuncts = Ifuncts if Ifuncts else []
        for ifunc in self.Ifuncts:
            ifunc(self.ifactory, self.max_str_len, self.chars)
        self.ghn.I = self.ifactory.I

        self.qfactory = QFactory(self.max_str_len, self.chars, self.conf, self.sets)
        self.Qfuncts = Qfuncts if Qfuncts else []
        if not self.Qfuncts:
            raise Warning("Qfuncts not defined- do you really want to train on empty connectivity matrix?")
        for qfunc in self.Qfuncts:
            qfunc(self.qfactory, self.max_str_len, self.chars)
        self.ghn.Q = self.qfactory.Q

        # Set losses, optimizer, and evaluator
        self.evaluator = evaluator if evaluator else AttributeError("Evaluator is not set.")
        self.clean_output = cleanfunct if cleanfunct else lambda s: s
        self.get_loss = loss if loss else EvalLoss(self.evaluator, self.max_str_len, self.num_syms, eval_clip=eval_clip, min_E=min_energy_for_eval)
        self.get_loss._set_tokenstring_preprocess_function(self.clean_output)
        self.get_loss._set_Q(self.ghn.Q)

        self.optimizer = optimizer
        if self.optimizer:
            self.optimizer.set_model(self)

        self.L_hist = [np.zeros_like(self.V)]

        self._set_internal_energy_domain()

    @property
    def u(self):
        return self.ghn.u

    @property
    def V(self):
        return self.ghn.V

    @property
    def Q(self):
        return self.ghn.Q

    @property
    def I(self):
        return self.ghn.I
    
    @property
    def nUnits(self):
        return self.ghn.nUnits
    
    def _set_internal_energy_domain(self):
        V_max, V_min = Heigen.find_extreme_eigenvectors(self.ghn.Q)
        self.ghn.V_extremes = (V_min.reshape((self.max_str_len * self.num_syms, 1)), V_max.reshape((self.max_str_len * self.num_syms, 1)))

        print(f"Q: {self.ghn.Q.shape}, V: {self.ghn.V_extremes[1].shape}")
        max_E = GHN.calc_energy_internal(self.ghn.Q, self.ghn.V_extremes[1])
        min_E = GHN.calc_energy_internal(self.ghn.Q, self.ghn.V_extremes[0])
        self.energy_domain = (min_E, max_E)
        print(self.energy_domain)
        self.get_loss._set_max_diff(self.ghn.V_extremes[1])

    #@flexmethod.staticsig('Q','u','V','E',eq=None)
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
        
    def update(self, I=None, n_iters=1000, min_dE=5, earlystopping=50, min_E=-130):
        """
        n_iters = epochs
        min_dE = minimum change in E before noticing change in E
        earlystopping = number of iters without change before quitting
        min_E = minimum energy before quitting
        """
        
        min_found_E = np.inf
        lastBigJump = 0

        for e in range(n_iters):
            
            # evallosses
            L = self.get_evalloss(eq = self.decode_output(V=self.V, clean=True))

            if self.optimizer is not None:
                L = self.optimizer.process(L, self. V)

            self.L_hist.append(L)

            s = GHN.forward(self.Q,self.u,self.V,L,dt=self.ghn.dt,gain=self.ghn.gain)
            self.ghn.update_from_state(s)

            if len(self.ghn.E_hist) > 1:
                dE = self.ghn.E_hist[-1] - self.ghn.E_hist[-2]

                if abs(dE) > min_dE:
                    lastBigJump = e
                min_found_E = min(min_found_E, s.E)
                
                if (e - lastBigJump) >= earlystopping:
                    break

                if min_E is not None and s.E < min_E:
                    if abs(len((self.V > 0.5)) - self.shape[0]) < 2:
                        break

    def plot_results(self):
        self.ghn.plot_V()

    def plot_Ehist(self, max_y=None):
        self.ghn.plot_Ehist(max_y)

    
    def plot_histories_as_video(self):
        
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))

        axes[0, 1].set_title('V')
        axes[0, 0].set_title('u')
        axes[1, 1].set_title('Eval Loss Matrix')
        axes[1, 0].set_title('Q@V Matrix')
        axes[0, 2].set_title('Energy')
        axes[1, 2].axis('off')


        #generate Q@V
        QV_hist = [self.Q@v for v in self.ghn.V_hist]

        # Reshape grid datas
        c_V_hist = [v.reshape(self.max_str_len, self.num_syms) for v in self.ghn.V_hist]
        c_u_hist = [u.reshape(self.max_str_len, self.num_syms) for u in self.ghn.u_hist]
        c_L_hist = [l.reshape((self.max_str_len, self.num_syms)) for l in self.L_hist]
        c_QV_hist = [qv.reshape((self.max_str_len, self.num_syms)) for qv in QV_hist]


        print(len(self.ghn.E_hist))
        print(len(c_QV_hist))
        print(self.ghn.E_hist)
        print(self.ghn.E_hist[0])

        # Get global min and max for consistent color limits
        global_min_V = min(matrix.min() for matrix in c_V_hist)
        global_max_V = max(matrix.max() for matrix in c_V_hist)
        global_min_u = min(matrix.min() for matrix in c_u_hist)
        global_max_u = max(matrix.max() for matrix in c_u_hist)
        global_min_L = min(matrix.min() for matrix in c_L_hist)
        global_max_L = max(matrix.max() for matrix in c_L_hist)
        global_min_QV = min(matrix.min() for matrix in c_QV_hist[5:])
        global_max_QV = max(matrix.max() for matrix in c_QV_hist[5:])

        im_V = axes[0, 1].imshow(c_V_hist[0], cmap='viridis', vmin=global_min_V, vmax=global_max_V)
        im_u = axes[0, 0].imshow(c_u_hist[0], cmap='viridis', vmin=global_min_u, vmax=global_max_u)
        im_L = axes[1, 1].imshow(c_L_hist[0], cmap='viridis', vmin=global_min_L, vmax=global_max_L)
        im_QV = axes[1, 0].imshow(c_QV_hist[0], cmap='viridis', vmin=global_min_QV, vmax=global_max_QV)
        line, = axes[0, 2].plot([], [])

        #fig.colorbar(im_V, ax=axes[0, 0])
        fig.colorbar(im_u, ax=axes[0, 0])
        fig.colorbar(im_L, ax=axes[1, 1])
        fig.colorbar(im_QV, ax=axes[1, 0])

        def animate(i):


            print(len(self.ghn.E_hist))
            print(len(c_QV_hist))
            print(self.ghn.E_hist)
            print(self.ghn.E_hist[0])


            s = self.decode_output(V=c_V_hist[i].reshape((self.nUnits,1)))
            y = self.evaluate(s)
            if y is None: y = "err"
            s += f" = {y}"
            fig.suptitle(f"{s}")
            
            im_V.set_array(c_V_hist[i])
            im_u.set_array(c_u_hist[i])
            im_L.set_array(c_L_hist[i])
            im_QV.set_array(c_QV_hist[i])
            print(i)
            
            if i >= 2:
                x_data = np.arange(2, i + 1)
                y_data = self.E_hist[2: i + 1]

                # Check if data arrays are not empty and have the same length
                if len(x_data) > 0 and len(y_data) > 0 and len(x_data) == len(y_data):
                    print(i)
                    print('success')
                    line.set_data(x_data, y_data)
                    #line.set_data(np.arange(1, i + 1), self.E_hist[1: i + 1])
                    axes[0, 2].relim()
                    axes[0, 2].autoscale_view(True, True, True)
                else:
                    print('oh no')
                    print(len(x_data))
                    print(x_data)
                    print(len(y_data))
                    print(y_data)
                    raise TypeError()


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
