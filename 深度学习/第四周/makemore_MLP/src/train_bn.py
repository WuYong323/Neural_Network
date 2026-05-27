from __future__ import annotations

import torch

from model import MakemoreMLP
from train import train,load_data,eval_loss


def main()->None:
    torch.manual_seed(42)
    steps = 50000
    batch_size = 32
    lr = 0.1
    block_size = 3
    embed_dim = 10
    hidden_size = 100

    Xtr, Ytr, Xdv, Ydv, Xte, Yte, _, _ = load_data(block_size=block_size)

    m1 = MakemoreMLP(block_size=block_size, embed_dim=embed_dim, hidden_size=hidden_size,use_bn=False)
    m2 = MakemoreMLP(block_size=block_size, embed_dim=embed_dim, hidden_size=hidden_size,use_bn=True)

    train(m1, Xtr, Ytr, Xdv, Ydv, steps=steps, batch_size=batch_size, lr=lr)
    train(m2, Xtr, Ytr, Xdv, Ydv, steps=steps, batch_size=batch_size, lr=lr)

    test_loss1 = eval_loss(m1, Xte, Yte)
    test_loss2 = eval_loss(m2, Xte, Yte)

    print(f"\nfinal test loss(no BN): {test_loss1:.4f}")
    print(f"\nfinal test loss(BN): {test_loss2:.4f}")


if __name__=="__main__":
    main()



















