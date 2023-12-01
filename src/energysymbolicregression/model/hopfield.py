import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, List, Tuple
from typing import Callable, List, Dict, Union
from hopfield_utils import *
from easytools.decorators import untypedmethod


# Simple struct for easily passing current state
class Hebbian_State:
    def __init__(self, u=None, Q=None, V=None, I=None, du=None, E=None):
        self.u = u
        self.Q = Q
        self.V = V
        self.I = I
        self.du = du
        self.E = E
    

# Graded Hopfield Network
class GHN:
    def __init__(self, Q, u_shape: Tuple[int, int] = None, u_init: np.ndarray = None, I: np.ndarray = None, gain: float = 999, dt: float = 0.01):

        if u_shape is None and u_init is None:
            raise AttributeError("No shape provided")
        if u_shape is None:
            u_shape == u_init.shape
        else:
            u_init = (np.random.rand(u_shape[0]*u_shape[1], 1) - 0.5) * 0.01 + 0.5
        if I is None:
            I = np.zeros((u_shape[0]*u_shape[1], 1))


        self.shape = u_shape
        self.nUnits = self.shape[0] * self.shape[1]
        self.gain = gain
        self.dt = dt
        self.u = u_init
        self.V = self.squasher(self.u, self.gain)
        self.E_hist = [np.inf]
        self.V_hist = [self.V]
        self.u_hist = [self.u]
        self.Q = Q
        self.I = I



    #----- Learning Functions ------#

    @staticmethod
    def squasher(u, gain):
        return 0.5 * (1 + np.tanh((u - 0.5) * gain))
    
    @staticmethod
    def hebbian_learning(u: np.ndarray, Q: np.ndarray, V: np.ndarray, I: np.ndarray, dt: float, gain: float) -> Hebbian_State:
        du = -u + Q @ V + I
        u += du * dt
        V = GHN.squasher(u, gain)

        return Hebbian_State(u, Q, V, I, du)
    
    def forward(self):
        s = GHN.hebbian_learning(self.u, self.Q, self.V, self.I, self.dt, self.gain)
        self.u = s.u
        self.V = s.V
        s.E = self.get_energy()

        self.u_hist.append(s.u)
        self.V_hist.append(s.V)
        self.E_hist.append(s.E)



    #----- Energy Functions -----#

    @untypedmethod('Q', 'V')
    def calc_energy_internal(self) -> float:
        return -0.5 * np.dot(self.V.T, np.dot(self.Q, self.V))[0][0]
    
    @untypedmethod("V", "I")
    def calc_energy_external(self) -> float:
        return -1 * np.dot(self.V.T, self.I)[0][0]
    
    @untypedmethod("Q", "V", "I")
    def calc_energy(self):
        return GHN.calc_energy_internal(self.Q, self.V) + GHN.calc_energy_external(self.V, self.I)


    #----- System State -----#

    @untypedmethod("Q")
    def find_best_eigenvector_by_energy(self):
        """
        
        Get eigenvector correlating with lowest internal energy by searching all eigenvectors. 
        
        """

        _, eigenvectors = np.linalg.eig(self.Q)
        min_e = np.inf
        min_i = -1
        for i in range(eigenvectors.shape[1]):
            v = eigenvectors[:,i].reshape((Q.shape[0], 1))
            e = GHN.calc_internal_energy(Q, v)[0][0]
            if e < min_e:
                min_i = i
                min_e = e

        v_min_e = eigenvectors[:,min_i].reshape((Q.shape[0], 1))
        return v_min_e
    
    @untypedmethod("Q")
    def calc_min_E(self):
        v_min = GHN.find_best_eigenvector_by_energy(self.Q)
        


    def _set_internal_energy_domain(self):

        V_max, V_min = find_extreme_eigenvectors(self.Q)

        self.V_extremes = (V_min.reshape((self.shape[0]*self.shape[1], 1)),
                           V_max.reshape((self.shape[0]*self.shape[1], 1)))
        
        max_E = GHN.calc_energy_internal(self.Q, self.V_extremes[1])
        min_E = GHN.calc_energy_internal(self.Q, self.V_extremes[0])

        self.energy_domain = (min_E, max_E)

        print(self.energy_domain)



    #----- Update -----#

    @untyped{}
    def update(self, n_iters=1000, min_dE=5, earlystopping=50, min_E=-130):
        """
        n_iters = epochs
        min_dl = minimum change in L before noticing change in L
        earlystopping = number of iters without change before quitting
        min_L = minimum energy before quitting
        """

        lastBigJump = 0

        for e in range(n_iters):

            s = self.forward()

            if len(self.E_hist) > 1:
                dE = self.E_hist[-1] - self.E_hist[-2]

                if abs(dE) > min_dE:
                    lastBigJump = e
                min_found_E = min(min_found_E, s.E)
                
                if (e - lastBigJump) >= earlystopping:
                    break

                if min_E is not None and s.E < min_E:
                    if abs(len((self.V > 0.5)) - self.shape[0]) < 2:
                        break



    #----- Visualization -----#

    def plot_V(self):
        V_reshaped = self.V.reshape(self.shape[0], self.shape[1])
        plt.imshow(V_reshaped)
        plt.show()

    def plot_u(self):
        u_reshaped = self.u.reshape(self.shape[0], self.shape[1])
        plt.imshow(u_reshaped)
        plt.show()

    def plot_Ehist(self, max_y=None):
        #plt.set_xlim((1,len(self.L_hist)))
        plt.plot(np.arange(1, len(self.E_hist)), self.E_hist[1:])
        if max_y is not None:
            plt.ylim((np.min(self.E_hist[1:])-5, min(np.max(self.E_hist[1:])+5,max_y)))#np.max(self.E_hist[1:])+5))

        plt.show()

    def plot_histories_as_video(self):

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        axes[0, 1].set_title('V')
        axes[0, 0].set_title('u')
        axes[1, 0].set_title('Q@V Matrix')
        axes[1, 1].set_title('Energy')

        #generate Q@V
        QV_hist = [self.Q@v for v in self.V_hist]

        # Reshape grid datas
        c_V_hist = [v.reshape(self.shape[0], self.shape[1]) for v in self.V_hist]
        c_u_hist = [u.reshape(self.shape[0], self.shape[1]) for u in self.u_hist]
        c_QV_hist = [qv.reshape((self.shape[0], self.shape[1])) for qv in QV_hist]

        # Get global min and max for consistent color limits
        global_min_V = min(matrix.min() for matrix in c_V_hist)
        global_max_V = max(matrix.max() for matrix in c_V_hist)
        global_min_u = min(matrix.min() for matrix in c_u_hist)
        global_max_u = max(matrix.max() for matrix in c_u_hist)
        global_min_QV = min(matrix.min() for matrix in c_QV_hist[5:])
        global_max_QV = max(matrix.max() for matrix in c_QV_hist[5:])

        im_V = axes[0, 1].imshow(c_V_hist[0], cmap='viridis', vmin=global_min_V, vmax=global_max_V)
        im_u = axes[0, 0].imshow(c_u_hist[0], cmap='viridis', vmin=global_min_u, vmax=global_max_u)
        im_QV = axes[1, 0].imshow(c_QV_hist[0], cmap='viridis', vmin=global_min_QV, vmax=global_max_QV)
        line, = axes[1, 1].plot([], [])

        #fig.colorbar(im_V, ax=axes[0, 0])
        fig.colorbar(im_u, ax=axes[0, 0])
        fig.colorbar(im_QV, ax=axes[1, 0])

        def animate(i):
            
            im_V.set_array(c_V_hist[i])
            im_u.set_array(c_u_hist[i])
            im_QV.set_array(c_QV_hist[i])
            
            if i >= 1:
                line.set_data(np.arange(1, i + 1), self.E_hist[1: i + 1])
                axes[1, 1].relim()
                axes[1, 1].autoscale_view(True, True, True)


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
