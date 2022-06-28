import torch
import torch.nn.functional as F
import numpy as np
import warnings


def similarity(vec1, vec2):
    if torch.linalg.norm(vec1) < 1e-8 or torch.linalg.norm(vec2) < 1e-8:
        warnings.warn("Small vector in similarity!")
    #     return torch.tensor(1.0, device=vec1.device)
    # else:
    vec1 = vec1.type(torch.float64)
    vec2 = vec2.type(torch.float64)
    # return F.cosine_similarity(vec1 / torch.linalg.norm(vec1), vec2 / torch.linalg.norm(vec2), dim=0).clip(-1.0, 1.0)
    return F.cosine_similarity(vec1 / torch.linalg.norm(vec1), vec2 / torch.linalg.norm(vec2), dim=0, eps=1e-20).clip(-1.0, 1.0)
    # return F.cosine_similarity(vec1, vec2, dim=0).clip(-1.0, 1.0)
    #return F.cosine_similarity(vec1.type(torch.float64), vec2.type(torch.float64), dim=0).clip(-1.0, 1.0)


def generate_pd_matrix(dim, device, dtype):
    from sklearn.datasets import make_spd_matrix
    m = torch.tensor(make_spd_matrix(dim)).to(device).type(dtype)
    return m


def generate_positive_full_rank_matrix(dim, device, dtype):
    import math
    m = torch.rand((dim,dim)).to(device).to(dtype)
    m /= torch.sum(m, dim=1).reshape(-1, 1)
    return m