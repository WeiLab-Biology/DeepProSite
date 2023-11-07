import torch
from torch import optim

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(task, parameters, d_model):
    train_size = {"PRO":335, "CA":1550, "MG":1729, "MN":547, "Metal":5469}
    batch_size = 32
    warmup_epoch = (5 if task != "Metal" else 2)
    step_each_epoch = int(train_size[task] / batch_size)
    warmup = warmup_epoch * step_each_epoch
    top_lr = 0.0004
    factor = top_lr / (d_model ** (-0.5) * min(warmup ** (-0.5), warmup * warmup ** (-1.5)))

    return NoamOpt(
        d_model, factor, warmup, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
