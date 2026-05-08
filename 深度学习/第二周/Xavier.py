import numpy as np


def initialize_parameters_xavier(layer_dims,seed=1):
    np.random.seed(seed)

    parameters={}
    L=len(layer_dims)

    for l in range(1,L):
        n_in=layer_dims[l-1]
        n_out=layer_dims[l]

        std=np.sqrt(2/(n_in+n_out))

        parameters[f"W{l}"]=np.random.randn(n_in,n_out)*std
        parameters[f"b{l}"]=np.zeros(n_out)

    return parameters

params=initialize_parameters_xavier([784,128,64,10])

for k,v in params.items():
    if k.startswith('W'):
        l=int(k[1:])
        n_in,n_out=v.shape
        expected_std=np.sqrt(2.0/(n_in+n_out))
        print(f"{k}: shape={v.shape}, "
              f"mean={v.mean():+.4f}, "
              f"std={v.std():.4f} (期望≈{expected_std:.4f})")
    else:
        print(f"{k}: shape={v.shape}, all_zero={(v == 0).all()}")