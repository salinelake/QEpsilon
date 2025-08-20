import numpy as np
import torch as th



## dynamical decoupling sequence
class XY8:
    def __init__(self, cycle_time: float):
        self.direction_sequence = [th.tensor([1.0,0,0]), th.tensor([0,1.0,0]), th.tensor([1.0,0,0]), th.tensor([0,1.0,0]), 
                                th.tensor([0,1.0,0]), th.tensor([1.0,0,0]), th.tensor([0,1.0,0]),  th.tensor([1.0,0,0]), ]
        self.phase_sequence = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
        self.idx = -1
        self.tau = cycle_time / 16
        self.t = - self.tau
    def __iter__(self):
        return self

    def __next__(self):
        self.idx = (self.idx + 1) % 8
        self.t += self.tau * 2
        direction = self.direction_sequence[self.idx]
        phase = self.phase_sequence[self.idx]
        return self.t, direction, phase

