from __future__ import annotations

import torch
from torch import Tensor, meshgrid

import matplotlib.pyplot as plt



def sgd_step(w,g,lr:float=0.1):
    return w-lr*g


def momentum_step(w,g,v,lr:float=0.1,beta:float=0.9):
    v=beta*v+g
    return w-lr*v,v

def rmsprop_step(w,g,s,lr:float=0.1,beta2:float=0.999,eps:float=1e-8):
    s=beta2*s+(1-beta2)*(g**2)
    return w-lr*g/(s.sqrt()+eps),s

def adam_step(w,g,m,v,t,lr:float=0.1,b1:float=0.9,b2:float=0.999,eps:float=1e-8):
    m=b1*m+(1-b1)*g
    v=b2*v+(1-b2)*(g**2)
    m_hat=m/(1-b1**t)
    v_hat=v/(1-b2**t)
    return w-lr*m_hat/(v_hat.sqrt()+eps),m,v


def loss_fn(w:Tensor)->Tensor:
    return 10.0*w[0]**2+0.5*w[1]**2



def run(name:str,n_step:int=80):
    w=torch.tensor([1.5,1.5],requires_grad=True)
    v=torch.zeros_like(w)
    s=torch.zeros_like(w)
    m=torch.zeros_like(w)
    traj=[w.detach().clone()]

    for t in range(1,n_step+1):
        loss=loss_fn(w)
        g,=torch.autograd.grad(loss,w)

        with torch.no_grad():
            if name=="SGD":
                w_new=sgd_step(w,g,lr=0.1)
            elif name=="Momentum":
                w_new,v=momentum_step(w,g,v,lr=0.1,beta=0.9)
            elif name=="RMSProp":
                w_new,s=rmsprop_step(w,g,s,lr=0.1)
            elif name=="Adam":
                w_new,m,v=adam_step(w,g,m,v,t,lr=0.1)
            w.copy_(w_new)
        traj.append(w.detach().clone())

    return torch.stack(traj).numpy()


def main()->None:
    fig,ax=plt.subplots(figsize=(12,8))

    xs=torch.linspace(-2,2,100)
    ys=torch.linspace(-2,2,100)
    X,Y=torch.meshgrid(xs,ys,indexing="xy")
    Z=10.0*X**2+0.5*Y**2
    ax.contour(X,Y,Z,levels=20,cmap="gray",alpha=0.5)

    for name, color in [("SGD", "tab:blue"), ("Momentum", "tab:orange"),
                        ("RMSProp", "tab:green"), ("Adam", "tab:red")]:
        traj=run(name)
        ax.plot(traj[:,0],traj[:,1],"-o",ms=3,color=color,label=name)

    ax.scatter([0],[0],marker="*",color="black",s=300,label="optimum")
    ax.legend()
    ax.set_title("Optimizer trajectories on ill-conditioned quadratic")
    ax.set_xlabel("w[0] (steep)")
    ax.set_ylabel("w[1] (flat)")

    plt.tight_layout()
    plt.savefig("logs/optim_compare.png",dpi=120)
    plt.show()



def animate_optimizers(n_step: int = 80, interval_ms: int = 80, save_gif: str | None = None):
    from matplotlib.animation import FuncAnimation

    names_colors = [("SGD", "tab:blue"), ("Momentum", "tab:orange"),
                    ("RMSProp", "tab:green"), ("Adam", "tab:red")]
    trajs = {name: run(name, n_step) for name, _ in names_colors}

    fig, ax = plt.subplots(figsize=(10, 8))
    xs = torch.linspace(-2, 2, 100)
    ys = torch.linspace(-2, 2, 100)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    Z = 10.0 * X ** 2 + 0.5 * Y ** 2
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap="gray", alpha=0.5)
    ax.scatter([0], [0], marker="*", color="black", s=200, label="optimum")

    lines, heads = {}, {}
    for name, color in names_colors:
        (line,) = ax.plot([], [], "-", color=color, alpha=0.6, label=name)
        (head,) = ax.plot([], [], "o", color=color, ms=7)
        lines[name], heads[name] = line, head

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("w[0] (steep)")
    ax.set_ylabel("w[1] (flat)")
    ax.set_title("Optimizer trajectories (animated)")
    ax.legend(loc="upper right")
    step_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    def update(frame):
        for name, _ in names_colors:
            t = trajs[name][: frame + 1]
            lines[name].set_data(t[:, 0], t[:, 1])
            heads[name].set_data([t[-1, 0]], [t[-1, 1]])
        step_text.set_text(f"step {frame}/{n_step}")
        return [*lines.values(), *heads.values(), step_text]

    anim = FuncAnimation(fig, update, frames=n_step + 1,
                         interval=interval_ms, blit=True, repeat=False)

    if save_gif:
        anim.save(save_gif, writer="pillow", fps=max(1, 1000 // interval_ms))
    else:
        plt.show()
    return anim


if __name__=="__main__":
    #main()
    #animate_optimizers()
    animate_optimizers(save_gif="logs/optim_anim.gif")



























