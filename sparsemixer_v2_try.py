import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsemixer import get_router

class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        scores: torch.Tensor, 
        multiplier: torch.Tensor, 
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one
        
    @staticmethod
    def backward(
        ctx, 
        grad_at_output: torch.Tensor, 
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors
        
        grad_at_output = grad_at_output * multiplier
        
        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )
        
        return (
            grad_at_scores_expaned, 
            None, 
            None, 
            None, 
            None, 
        )

def sparsemixer(scores, top_k, jitter_eps, training):
    assert top_k == 2
    
    ################ first expert ################
    
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts = (
            masked_gates - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
    else:
        selected_experts = max_ind
        
    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
    
    if training:
        # compute midpoint mask 
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        ) 
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier = mp.apply(
            scores, 
            multiplier_o, 
            selected_experts, 
            masked_gates, 
            mask_for_one,
        )
    else:
        multiplier = multiplier_o

    # masked out first expert 
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts_top2 = (
            masked_gates_top2 - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
    else:
        selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
    
    if training: 
        # compute midpoint mask 
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        ) 
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2 = mp.apply(
            scores, 
            multiplier_top2_o, 
            selected_experts_top2, 
            masked_gates_top2, 
            mask_for_one_top2,
        )
    else:
        multiplier_top2 = multiplier_top2_o
    
    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)
        
    return (
        multiplier, 
        selected_experts,
    )

n_input = 100  # Number of input
x_dim = 2048  # Input didden dimension
h_dim = 512  # Hidden dimension
E = 6  # Number of expert
top_k = 2
dtype = torch.bfloat16

##### Router config:
training = True
jitter_eps = 0.1

logits = torch.randn(n_input, E, dtype=dtype)

multiplier, selected_experts = sparsemixer(logits, top_k, jitter_eps, training = training)

print("####### multiplier #######")
print(multiplier.shape)
print(multiplier)
print("####### selected_experts #######")
print(selected_experts.shape)
print(selected_experts)