#
# LLM disclosure: AI generated benchmark
#

import math
import time

import torch
from torch.optim import Optimizer
from torch.profiler import ProfilerActivity, profile


class AdamWNaive(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                alpha = group["lr"]
                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # ### DIFFERENT ###
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                # ### END DIFFERENT ###

                alpha_t = alpha
                if group["correct_bias"]:
                    alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # ### DIFFERENT ###
                p.data -= alpha_t * exp_avg / (exp_avg_sq.sqrt() + eps)

                if weight_decay != 0.0:
                    p.data -= alpha * weight_decay * p.data
                # ### END DIFFERENT ###

        return loss


class AdamWInplace(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                alpha = group["lr"]
                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # ### DIFFERENT ###
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # ### END DIFFERENT ###

                alpha_t = alpha
                if group["correct_bias"]:
                    alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # ### DIFFERENT ###
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-alpha_t)

                if weight_decay != 0.0:
                    p.data.mul_(1 - alpha * weight_decay)
                # ### END DIFFERENT ###

        return loss


VARIANTS = [("naive", AdamWNaive), ("inplace", AdamWInplace)]

SMALL = [(1024, 256), (256,), (256, 256), (256,), (256, 10), (10,)]


def gpt2_small():
    d = 768
    vocab = 50257
    n_pos = 1024
    n_layer = 12

    shapes = [(vocab, d), (n_pos, d)]
    for _ in range(n_layer):
        shapes += [(d, d), (d,)] * 4          # q, k, v, attn out proj
        shapes += [(4 * d, d), (4 * d,)]      # mlp in
        shapes += [(d, 4 * d), (d,)]          # mlp out
        shapes += [(d,)] * 4                  # two layernorms
    shapes += [(d,)] * 2                      # final layernorm
    return shapes


CASES = [("small", SMALL, 200), ("gpt2-small", gpt2_small(), 10)]


def make_params(shapes, device, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = []
    for shape in shapes:
        p = torch.nn.Parameter((torch.randn(shape, generator=g, dtype=dtype) * 0.02).to(device))
        p.grad = (torch.randn(shape, generator=g, dtype=dtype) * 0.01).to(device)
        params.append(p)
    return params


def refill_grads(params, seed):
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad.copy_((torch.randn(p.shape, generator=g) * 0.01).to(p.device))


def check_equivalence(device, steps=50, dtype=torch.float64):
    final = {}
    for name, cls in VARIANTS:
        params = make_params(SMALL, device, dtype, seed=1)
        opt = cls(params, lr=1e-3, weight_decay=0.01)
        for t in range(steps):
            refill_grads(params, seed=100 + t)
            opt.step()
        final[name] = [p.detach().clone() for p in params]

    diff = max((a - b).abs().max().item()
               for a, b in zip(final["naive"], final["inplace"]))
    scale = max(a.abs().max().item() for a in final["naive"])
    status = "OK" if diff / scale < 1e-10 else "FAIL"
    print(f"equivalence ({steps} steps, {dtype}): rel diff {diff / scale:.2e}  {status}")


def alloc_bytes(opt, device):
    if device.type == "cuda":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        attr = "self_device_memory_usage"
    else:
        activities = [ProfilerActivity.CPU]
        attr = "self_cpu_memory_usage"

    with profile(activities=activities, profile_memory=True) as prof:
        opt.step()
    return sum(max(getattr(e, attr, 0), 0) for e in prof.key_averages())


def time_step(opt, iters, device, warmup=5):
    for _ in range(warmup):
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            opt.step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.step()
    return (time.perf_counter() - t0) / iters * 1e3


def run(device):
    print(f"\n=== {device} ===")
    check_equivalence(device)

    for label, shapes, iters in CASES:
        nparams = sum(math.prod(s) for s in shapes)
        nbytes = 4 * nparams
        print(f"\n{label}: {nparams / 1e6:.2f}M params, "
              f"{len(shapes)} tensors, {nbytes / 2**20:.1f} MiB")
        print(f"  {'variant':8s} {'ms/step':>9s} {'speedup':>8s} "
              f"{'alloc/step':>11s} {'x params':>9s}")

        base = None
        for name, cls in VARIANTS:
            params = make_params(shapes, device)
            opt = cls(params, lr=1e-5, weight_decay=0.01)
            ms = time_step(opt, iters, device)
            allocated = alloc_bytes(opt, device)
            base = ms if base is None else base
            print(f"  {name:8s} {ms:9.2f} {base / ms:7.2f}x "
                  f"{allocated / 2**20:8.1f} MiB {allocated / nbytes:8.2f}x")

            del params, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(1)
    run(torch.device("cpu"))
    if torch.cuda.is_available():
        run(torch.device("cuda"))
