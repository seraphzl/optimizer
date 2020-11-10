# Reference:
#   https://arxiv.org/abs/1908.03265
#   https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py
#   https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW

import math
import torch
from torch.optim.optimizer import Optimizer
from gCentralization import centralized_gradient


class RAdamgc(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, degenerated_to_sgd=False,
                 use_gc=False, gc_conv_only=False, gc_loc=False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None]]*10
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        buffer=[[None, None, None]]*10)
        super(RAdamgc, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdamgc, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                m_t, v_t = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    v_t_max = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                # exp_avg
                m_t.mul_(beta1).add_(grad, alpha=1-beta1)
                # exp_avg_sq
                v_t.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(v_t_max, v_t, out=v_t_max)

                state['step'] += 1
                beta1_t = beta1**state['step']
                beta2_t = beta2**state['step']

                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    rho_t, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    rho_max = 2/(1-beta2) - 1
                    rho_t = rho_max - 2*state['step']*beta2_t/(1-beta2_t)
                    buffered[1] = rho_t

                    # more conservative since it's an approximated value
                    if rho_t >= 5:
                        step_size = math.sqrt((1-beta2_t)*(rho_t-4)/(rho_max-4)*(rho_t-2)/(rho_max-2)*rho_max/rho_t)/(1-beta1_t)
                    elif self.degenerated_to_sgd:
                        step_size = 1.0/(1-beta1_t)
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if rho_t >= 5:
                    if amsgrad:
                        G_grad = torch.mul(p_data_fp32, group['weight_decay']).addcdiv_(m_t, v_t_max.sqrt().add_(group['eps']), value=step_size)
                    else:
                        G_grad = torch.mul(p_data_fp32, group['weight_decay']).addcdiv_(m_t, v_t.sqrt().add_(group['eps']), value=step_size)
                    if self.gc_loc == False:
                        G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                    p_data_fp32.add_(G_grad, alpha=-group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    G_grad = torch.mul(p_data_fp32, group['weight_decay']).add_(m_t, alpha=step_size)
                    if self.gc_loc == False:
                        G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                    p_data_fp32.add_(G_grad, alpha=-group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss
