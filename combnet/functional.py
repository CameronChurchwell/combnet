from combnet.filters import *
import torch

# Apply a multiple on multiple inputs pairwise
paired_comb = torch.vmap( single_fractional_comb_fiir, in_dims=(0, 0, 0, None))
# paired_comb = torch.vmap( fractional_comb_fiir, in_dims=(0, 0, 0, None))

# Make a "dot product" of the above
dot_comb = lambda x,f0,a,sr: paired_comb( x, f0, a, sr).sum(0)

# Use multiple filter sets on the above
multi_comb = torch.vmap( dot_comb, in_dims=(None, 0, 0, None))

# Add a batch dimension to the above
batch_comb = torch.vmap( multi_comb, in_dims=(0, None, None, None))

# Make a "linear combination" of paired_comb
lc_comb = lambda x,f0,a,sr,g=torch.tensor(1.0): (g[..., None] * paired_comb(x, f0, a, sr)).sum(0)

# Use multiple filter sets with lc_comb
lc_multi_comb = torch.vmap( lc_comb, in_dims=(None, 0, 0, None, 0))

# Make a batched version of above
lc_batch_comb = torch.vmap( lc_multi_comb, in_dims=(0, None, None, None, None))
