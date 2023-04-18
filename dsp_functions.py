import numpy as np
from scipy import signal
import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.

    Returns:
    A tensor with pointwise nonlinearity applied.
    """
    return max_value * jax.nn.sigmoid(x)**jnp.log(exponent) + threshold

def specplot(out,SR,fig_size=(6,2)):
    f, t, Sxx = signal.spectrogram(out, SR)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.pcolormesh(t, f, Sxx, shading='gouraud')
    
    return fig


