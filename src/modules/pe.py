import torch
import torch.amp as amp
def sinusoidal_embedding_1d(dim, position, device='cuda'):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=device).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast('cuda', enabled=False)
def rope_apply(x, freqs):
 
    b, s, n, c = x.shape
    # precompute multipliers
    x = torch.view_as_complex(x.to(torch.float64).reshape(
        b, s, n, -1, 2))
    # apply rotary embedding
    x = torch.view_as_real(x * freqs).flatten(3)
    return x.float()


@amp.autocast('cuda', enabled=False)
def get_seq_rope(head_dim, grid_sizes, freqs):
    c = head_dim // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        # precompute multipliers
        
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        output.append(freqs_i)

    return torch.cat(output, dim=0)[None,...]



@amp.autocast('cuda', enabled=False)
def get_seq_rope_causal(head_dim, grid_sizes, freqs, start):
    c = head_dim // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        freqs_i = torch.cat([
            freqs[0][start:f+start].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        output.append(freqs_i)
    #import tensorpc
    #tensorpc.dbg.breakpoint()
    return torch.cat(output, dim=0)[None,...]
