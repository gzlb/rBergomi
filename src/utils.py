import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass

@dataclass
class RBergomiUtils:
    a: float

    def g(self, x:float) -> float:
        """
        TBSS kernel applicable to the rBergomi variance process.
        """
        return x ** self.a

    def b(self, k:float) -> float:
        """
        Optimal discretisation of TBSS process for minimizing hybrid scheme error.
        """
        return ((k ** (self.a + 1) - (k - 1) ** (self.a + 1)) / (self.a + 1)) ** (1 / self.a)

    @staticmethod
    def cov(a, n:int) -> np.ndarray:
        """
        Covariance matrix for given alpha and n, assuming kappa = 1 for tractability.
        """
        cov = np.array([[0., 0.], [0., 0.]])
        cov[0, 0] = 1. / n
        cov[0, 1] = 1. / ((1. * a + 1) * n ** (1. * a + 1))
        cov[1, 1] = 1. / ((2. * a + 1) * n ** (2. * a + 1))
        cov[1, 0] = cov[0, 1]
        return cov

    @staticmethod
    def bs(F: float, K: float, V: float, o: str) -> float:
        """
        Returns the Black call price for given forward, strike, and integrated variance.
        """
        option_mapping = {'call': 1, 'put': -1, 'otm': 2 * (K > 1.0) - 1}
        
        w = option_mapping.get(o, 0)

        sv = np.sqrt(V)
        d1 = np.log(F / K) / sv + 0.5 * sv
        d2 = d1 - sv
        P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
        return P

    @staticmethod
    def bsinv(P: float, F: float, K: float, t: float, o: str) -> float:
        """
        Returns implied Black vol from the given call price, forward, strike, and time to maturity.
        """
        option_mapping = {'call': 1, 'put': -1, 'otm': 2 * (K > 1.0) - 1}
        
        w = option_mapping.get(o, 0)

        # Ensure at least intrinsic value
        P = np.maximum(P, np.maximum(w * (F - K), 0))

        def error(s):
            return RBergomiUtils.bs(F, K, s ** 2 * t, o) - P

        s = brentq(error, 1e-9, 1e+9)
        return s


