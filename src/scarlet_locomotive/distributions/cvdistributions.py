import secrets
import math

def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    #53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)  # in [0, 1)
    return a + (b - a) * u


def exponentialdist(lam):
    #step a: get a number from U(0, 1)
    y = uniform()

    #can't take ln(0), so get a new number if y is 0
    while y == 0:
        y = uniform()

    #step b: x = -(1/lambda) * ln(y)
    x = -(1 / lam) * math.log(y)
    return x
