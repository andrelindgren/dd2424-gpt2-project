from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # NOTE: I learned in Language Engineering course
                # that one should not create new temporary tensors
                # but instead modify the existing ones in order to save memory.
                # So it is a bit hard to read, but i kept the original version as well as comments.

                # Names used in params
                # t = step
                # m = exp_avg
                # v = exp_avg_sq

                # t = 0
                # m_0 = 0
                # v_0 = 0
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # m, v = state["exp_avg"], state["exp_avg_sq"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # t = t + 1
                state["step"] += 1
                t = state["step"]

                # 1. Update the first and second moments of the gradients.
                # m_t = beta1*m_{t-1} + (1-beta1)*g_t
                # state["exp_avg"] = m = beta1 * m + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
                # state["exp_avg_sq"] = v = beta2 * v + (1 - beta2) * grad * grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2. Apply efficient bias correction
                # alpha_t = alpha * sqrt(1-beta2^t)/(1-beta1*t)
                # I assume we will always use correct_bias = True, but might change later
                alpha_t = alpha
                if group["correct_bias"]:
                    alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 3. Update parameters (p.data).
                # theta_t = theta_{t-1} - alpha_t * m_t / ((sqrt(v_t) + eps))
                # p.data -= alpha_t * m / (v.sqrt() + eps)
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-alpha_t)

                # 4. Apply weight decay after the main gradient-based updates.
                if weight_decay != 0.0:
                    # Deviation from paper https://arxiv.org/pdf/1711.05101
                    # Alg. 2 row 12 have weight decay term be:  theta_t = theta_t − eta_t * lambda * theta_t
                    # But because group["lr"] = alpha * eta_t
                    # We must also multiply the weight decay with alpha so they match
                    #
                    # theta_t = theta_t − alpha * lambda * theta_t
                    # p.data -= alpha * weight_decay * p.data
                    p.data.add_(p.data, alpha=-alpha * weight_decay)

        return loss
