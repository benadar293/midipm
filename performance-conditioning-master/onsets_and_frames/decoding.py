import numpy as np
import torch


def extract_notes(onsets, frames, velocity,
                  onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    # onsets_forward = torch.roll(onsets, shifts=(1, 0), dims=(0, 1))
    # onsets_forward[0, :] = 0
    # onsets_backward = torch.roll(onsets, shifts=(-1, 0), dims=(0, 1))
    # onsets_backward[-1, :] = 0
    # onsets_peak = torch.logical_and(onsets >= onsets_forward, onsets >= onsets_backward)
    # onsets_peak = torch.logical_and(onsets >= 0.25, onsets_peak)

    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
    # onset_diff = torch.cat([frames[:1, :], frames[1:, :] - frames[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    # for nonzero in onsets_peak.nonzero(as_tuple=False):
    for nonzero in onset_diff.nonzero(as_tuple=False):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
            # if frames[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape, mask=None):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        # print('pitch', pitch, onset, offset)
        # print('onset offset', onset, offset, pitch)
        roll[onset: offset, pitch] = 1
    if mask is not None:
        roll *= mask
    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    # if mask_size is not None:
    #     mask = np.zeros(tuple(shape))
    #     notes = roll.shape[1]
    #     for n in range(notes):
    #         onset_d = roll[1:, n] - roll[: -1, n]
    #         print('unique', np.unique(onset_d))
    #         onset_d[onset_d < 0] = 0
    #         print('n', n, onset_d.sum())
    #         onset_d = np.concatenate((np.zeros((1, 1)), roll[1:, n] - roll[: -1, n]))
    #         onset_d[onset_d < 0] = 0
    #         for r in range(mask_size):
    #             mask[:, n] += np.roll(onset_d, r)
    return time, freqs
