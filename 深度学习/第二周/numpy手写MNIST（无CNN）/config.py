import numpy as np
import random

config={
    "seed":42,
    "batch":128,
    "lr":0.2,
    "l2":1e-4,
    "epochs":10,
    "lr_decay":10,
    "layer_dims":[784,256,10],
    "tag":"run"
}