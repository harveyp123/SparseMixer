import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsemixer import get_router

n_input = 100  # Number of input
x_dim = 2048  # Input didden dimension
h_dim = 512  # Hidden dimension
E = 64  # Number of expert
k = 6  # top k k value
dtype = torch.bfloat16

##### Router config:
routers= ['sparsemixer', 'switchgate']
router = routers[0]
load_balancing = True
jitter_eps = 0

logits = torch.randn(n_input, E, dtype=dtype)

router_sel = get_router(router)(E, x_dim, load_balancing, jitter_eps)

sample, multiplier, balance_loss = router_sel(logits)
print(sample)
print(multiplier)
print(balance_loss)