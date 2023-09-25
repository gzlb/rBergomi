import numpy as np
from utils import RBergomiUtils  
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class CorrelatedBrownian:
    dW1: np.ndarray
    dW2: np.ndarray

@dataclass
class PricePath:
    S: np.ndarray
    V: np.ndarray

class AbstractRBergomi(ABC):
    def __init__(self, n:int, N:int, T:float, a:float, rho:float):
        self.n = n  # Granularity (steps per year)
        self.N = N  # Paths
        self.T = T  # Maturity
        self.a = a  # Alpha
        self.rho = rho  # Correlation parameter
        self.dt = 1.0 / self.n  # Step size
        self.s = int(self.n * self.T)  # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[:, np.newaxis]  # Time grid

    @property
    @abstractmethod
    def e(self):
        pass

    @property
    @abstractmethod
    def c(self):
        pass

    @abstractmethod
    def generate_correlated_brownian(self) -> CorrelatedBrownian:
        pass

    @abstractmethod
    def calculate_price_path(self, dW: CorrelatedBrownian) -> PricePath:
        pass

class RBergomi(AbstractRBergomi):
    def e(self):
        return np.array([0, 0])

    def c(self):
        return RBergomiUtils.cov(self.a, self.n)  # Use RBergomiUtils for cov

    def generate_correlated_brownian(self) -> CorrelatedBrownian:
        rng = np.random.multivariate_normal
        dW1 = rng(self.e, self.c(), (self.N, self.s))  # Use self.c() to get covariance
        dW2 = np.random.randn(self.N, self.s) * np.sqrt(self.dt)
        return CorrelatedBrownian(dW1, dW2)

    def calculate_price_path(self, dW: CorrelatedBrownian) -> PricePath:
        dB = self.calculate_dB(dW)
        Y = self.calculate_Y(dW.dW1)
        V = self.calculate_V(Y)
        S = self.calculate_S(V, dB)
        return PricePath(S, V)

    def calculate_dB(self, dW: CorrelatedBrownian) -> np.ndarray:
        return self.rho * dW.dW1[:, :, 0] + np.sqrt(1 - self.rho**2) * dW.dW2

    def calculate_Y(self, dW1: np.ndarray) -> np.ndarray:
        Y1 = dW1[:, :, 1].copy()  # Assumes kappa = 1
        G = [RBergomiUtils.g(RBergomiUtils.b(k, self.a) / self.n, self.a) for k in range(2, 1 + self.s)]
        X = dW1[:, :, 0]  # Xi
        GX = np.array([np.convolve(G, X[i, :]) for i in range(self.N)])
        Y2 = GX[:, :1 + self.s]
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def calculate_V(self, Y: np.ndarray) -> np.ndarray:
        t = self.t.flatten()
        xi = 1.0
        eta = 1.0
        return xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * self.a + 1))

    def calculate_S(self, V: np.ndarray, dB: np.ndarray, S0=1) -> np.ndarray:
        self.S0 = S0
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * self.dt
        integral = np.cumsum(increments, axis=1)
        S = np.column_stack([S0 * np.exp(integral[i, :]) for i in range(self.N)])
        return S


