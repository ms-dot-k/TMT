"""
Useful learning rate schedulers
Warmup
CosineAnnealingLRWarmup
"""
import torch
import math
import functools


def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def LinearWarmup(optimizer, T_warmup):
    _decay_func = functools.partial(
        _constant_warmup,
        warmup_iterations=T_warmup
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


if __name__ == "__main__":

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Dummy parameters
    parameters = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    total_iters = 2000
    warmup_iters = 100

    # Test CosineAnnealingLRWarmup
    optimizer = torch.optim.Adam([parameters], lr=0.2)
    scheduler = CosineAnnealingLRWarmup(optimizer, T_max=total_iters, T_warmup=warmup_iters)
    actual_lr = []
    for _iter in range(total_iters):
        scheduler.step()
        actual_lr.append(optimizer.param_groups[0]["lr"])
    plt.plot(list(range(total_iters)), actual_lr, label="CosineAnnealingLRWarmup")
    plt.show()

    # Test LinearWarmup
    optimizer = torch.optim.Adam([parameters], lr=0.2)
    scheduler = LinearWarmup(optimizer, T_warmup=warmup_iters)
    actual_lr = []
    for _iter in range(total_iters):
        scheduler.step()
        actual_lr.append(optimizer.param_groups[0]["lr"])
    plt.plot(list(range(total_iters)), actual_lr, '--', label="LinearWarmup")
    plt.show()

    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.legend()
    plt.savefig("scheduler.png")