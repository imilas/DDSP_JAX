import tensorflow as tf

import numpy as np
import jax
import jax.numpy as jnp

import flax
from flax import linen as nn
from flax.training import train_state 
import optax

from audax.core import functional
from functools import partial

import dsp_functions as DSP


class SimpleClassifier(nn.Module):
    num_layers : int
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons
    @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, f0):
        x = nn.Dense(features=self.num_hidden//10)(f0)
        for i in range(self.num_layers):
            x = nn.Dense(features=self.num_hidden//(i+1))(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return jnp.abs(x)

class Decoder(nn.Module):
    def setup(self):
        lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                           variable_broadcast="params",
                           split_rngs={"params": False},
                           in_axes=1, 
                           out_axes=1,
                           length=250,
                           reverse=False)
        self.lstm1 = lstm_layer()
        self.dense1 = nn.Dense(100)
    @nn.remat
    def __call__(self,x):
        carry, hidden = arry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x),), size=100)
        (carry, hidden), x = self.lstm1((carry, hidden), x)
        x = self.dense1(x)
        x = nn.relu(x)
        return x

    
class harmonic_synthesizer():
    def __init__(self,SR,seconds=1):
        self.SR = SR
        self.seconds = seconds
        
    def make_sound(self,synth_params,f0,num_harmonics=10):
        dx = 1/(self.SR*self.seconds)
        angle = f0.cumsum() * dx * 2 * jnp.pi
        harmonic_angles = angle * jnp.arange(1,num_harmonics+1)[:,jnp.newaxis]
        harmonics = jnp.sin(harmonic_angles) * synth_params[:,jnp.newaxis]
        output = harmonics.sum(axis=0) # combine all sines
        return output
    
