import pickle
import torch
import numpy as np

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def ts2np(xs):
    if isinstance(xs, dict):
        out = {}
        for key in xs.keys():
            out[key] = ts2np(xs[key])
    elif isinstance(xs, list):
        out = []
        for x in xs:
            out.append(ts2np(x))
    elif isinstance(xs, torch.Tensor):
        out = xs.detach().cpu().numpy()
    else:
        out = xs
    return out


def np2ts(xs, device='cpu'):
    if isinstance(xs, dict):
        out = {}
        for key in xs.keys():
            out[key] = np2ts(xs[key], device)
    elif isinstance(xs, list):
        out = []
        for x in xs:
            out.append(np2ts(x), device)
    elif isinstance(xs, np.ndarray):
        out = torch.from_numpy(xs).to(device)
    else:
        out = xs
    return out