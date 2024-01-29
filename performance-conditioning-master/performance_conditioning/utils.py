import torch
from.constants import *
import numpy as np
import sys
from functools import reduce
from torch.nn.modules.module import _addindent


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def shift_label(label, shift):
    if shift == 0:
        return label
    assert len(label.shape) == 2
    t, p = label.shape
    keys, instruments = N_KEYS, p // N_KEYS
    label_zero_pad = torch.zeros(t, instruments, abs(shift), dtype=label.dtype)
    label = label.reshape(t, instruments, keys)
    to_cat = (label_zero_pad, label[:, :, : -shift]) if shift > 0 \
        else (label[:, :, -shift:], label_zero_pad)
    label = torch.cat(to_cat, dim=-1)
    return label.reshape(t, p)


def get_peaks(notes, win_size, gpu=False):
    constraints = []
    notes = notes.cpu()
    for i in range(1, win_size + 1):
        forward = torch.roll(notes, i, 0)
        forward[: i, ...] = 0  # assume time axis is 0
        backward = torch.roll(notes, -i, 0)
        backward[-i:, ...] = 0
        constraints.extend([forward, backward])
    res = torch.ones(notes.shape, dtype=bool)
    for elem in constraints:
        res = res & (notes >= elem)
    return res if not gpu else res.cuda()


def get_diff(notes, offset=True):
    rolled = np.roll(notes, 1, axis=0)
    rolled[0, :] = 0
    # return (rolled & (~notes)) if offset else (notes & (~rolled))
    return rolled > notes if offset else notes > rolled



def compress_across_octave(notes):
    keys = (MAX_MIDI - MIN_MIDI + 1)
    time, instruments = notes.shape[0], notes.shape[1] // keys
    notes_reshaped = notes.reshape((time, instruments, keys))
    notes_reshaped = notes_reshaped.max(axis=1)
    octaves = keys // 12
    res = np.zeros((time, 12), dtype=np.uint8)
    for i in range(octaves):
        curr_octave = notes_reshaped[:, i * 12: (i + 1) * 12]
        res = np.maximum(res, curr_octave)
    return res


def compress_time(notes, factor):
    t, p = notes.shape
    res = np.zeros((t // factor, p), dtype=notes.dtype)
    for i in range(t // factor):
        res[i, :] = notes[i * factor: (i + 1) * factor, :].max(axis=0)
    return res


def get_matches(index1, index2):
    matches = {}
    for i1, i2 in zip(index1, index2):
        # matches[i1] = matches.get(i1, []) + [i2]
        if i1 not in matches:
            matches[i1] = []
        matches[i1].append(i2)
    return matches


'''
Extend a temporal range to WINDOW_SIZE_SRC if it is shorter than that.
WINDOW_SIZE_SRC defaults to 28 frames for 256 hop length (assuming DTW_FACTOR=3), which is ~0.5 second.
'''
def get_margin(t_sources, max_len, WINDOW_SIZE_SRC = 11 * (512 // HOP_LENGTH) + 2 * DTW_FACTOR, min_left=0):
    margin = max(0, (WINDOW_SIZE_SRC - len(t_sources)) // 2)
    t_sources_left = list(range(max(t_sources[0] - margin, min_left), t_sources[0]))
    t_sources_right = list(range(t_sources[-1], min(t_sources[-1] + margin, max_len - 1)))
    t_sources_extended = t_sources_left + t_sources + t_sources_right
    return t_sources_extended


def get_inactive_instruments(target_onsets, T):
    keys = (MAX_MIDI - MIN_MIDI + 1)
    time, instruments = target_onsets.shape[0], target_onsets.shape[1] // keys
    notes_reshaped = target_onsets.reshape((time, instruments, keys))
    active_instruments = notes_reshaped.max(axis=(0, 2))
    res = np.zeros((T, instruments, keys), dtype=np.bool)
    for ins in range(instruments):
        if active_instruments[ins] == 0:
            res[:, ins, :] = 1
    return res.reshape((T, instruments * keys)), active_instruments


def max_inst(probs, thr=0.5, inactive=None, max_over_insts=True, max_pitch_inst=True):
    keys = MAX_MIDI - MIN_MIDI + 1
    instruments = probs.shape[1] // keys
    time = len(probs)
    probs = probs.reshape((time, instruments, keys))
    notes = (probs.max(axis=1) if max_over_insts else probs[:, -1, :]) >= thr
    if inactive:
        print('max inst inactive', inactive)
        for r_i in inactive:
            probs[:, r_i, :] = 0
    max_instruments = np.argmax(probs[:, : -1, :], axis=1)
    res = np.zeros(probs.shape, dtype=np.uint8)
    for t, p in zip(*(notes.nonzero())):
        res[t, max_instruments[t, p], p] = 1
        res[t, -1, p] = 1
    if max_pitch_inst:
        res = np.maximum(res, probs >= thr)
    return res.reshape((time, instruments * keys))

def get_rms(sig, normalized=True):
    assert len(sig.shape) == 1
    print('get rms', abs(sig).max(), (sig ** 2).max(), (sig ** 2).mean())
    return np.sqrt(((sig / 32768.) ** 2).mean()) if not normalized else np.sqrt((sig ** 2).mean())

def rms_normalize(sig, r=0.04):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    # linear rms level and scaling factor
    # r = 10 ** (rms_level / 20.0)
    print('rms before', get_rms(sig))
    a = np.sqrt((len(sig) * (r ** 2)) / np.sum(sig ** 2))
    # sig *= a
    sig = np.clip(sig * a, -1, 1) #* 32768.
    print("rms after", get_rms(sig))

    return sig

def map_performance(number):
    if number in list(range(1727, 1731)):
        return 0
    if number in list(range(1788, 1792)):
        return 1
    if number in list(range(1792, 1808)):
        return 2
    if number in list(range(1811, 1814)):
        return 3
    if number in list(range(1817, 1820)):
        return 4
    if number in list(range(1822, 1825)):
        return 5
    if number in list(range(1828, 1830)):
        return 6
    if number in list(range(1835, 1860)):
        return 7
    if number in list(range(1872, 1894)):
        return 8


    # if number in list(range(1916, 1934)):
    #     return 5
    # if number in list(range(2075, 2084)):
    #     return 6
    # if number in list(range(2104, 2107)):
    #     return 7
    # if number in list(range(2112, 2115)):
    #     return 8
    # if number in list(range(2116, 2120)):
    #     return 9
    # if number in list(range(2127, 2132)):
    #     return 10
    # if number in list(range(2148, 2152)):
    #     return 11
    # if number in list(range(2154, 2158)):
    #     return 12
    # if number in list(range(2158, 2162)):
    #     return 13
    # if number in list(range(2166, 2170)):
    #     return 14
    # if number in list(range(2177, 2181)):
    #     return 15
    # if number in list(range(2186, 2192)):
    #     return 16
    # if number in list(range(2202, 2205)):
    #     return 17
    # if number in list(range(2217, 2223)):
    #     return 18
    # if number in list(range(2241, 2245)):
    #     return 19
    # if number in list(range(2282, 2286)):
    #     return 20
    # if number in list(range(2288, 2290)):
    #     return 21
    # if number in list(range(2293, 2299)):
    #     return 22
    # if number in list(range(2313, 2316)):
    #     return 23
    # if number in list(range(2318, 2321)):
    #     return 24
    # if number in list(range(2330, 2343)):
    #     return 25
    # if number in list(range(2376, 2385)):
    #     return 26
    # if number in list(range(2397, 2399)):
    #     return 27
    # if number in list(range(2415, 2418)):
    #     return 28
    # if number in list(range(2431, 2434)):
    #     return 29
    # if number in list(range(2462, 2467)):
    #     return 30
    # if number in list(range(2480, 2484)):
    #     return 31
    # if number in list(range(2504, 2508)):
    #     return 32
    # if number in list(range(2521, 2524)):
    #     return 33
    # if number in list(range(2570, 2574)):
    #     return 34
    # if number in list(range(2626, 2630)):
    #     return 35
    raise ValueError

# def extract_inst(tensor, j, dim=2, n_insts=2):
#     old_dims = tensor.shape
#     new_dims = torch.cat((old_dims[: dim]), torch.Tensor((n_insts, N_KEYS)), old_dims[dim + 1:])
#     return tensor.view(new_dims)