from .constants import *
import numpy as np
from dtw import *
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import os
# from onsets_and_frames.mel import melspectrogram

from datetime import datetime
from onsets_and_frames.midi_utils import *
from onsets_and_frames.utils import *
import time
import fluidsynth
import librosa
from glob import glob
from midi2tsv_musicnet_multi import parse_midi_multi


def pool_k(x, k):
    if x.shape[0] % 2 != 0:
        x = torch.cat((x, torch.zeros((1, x.shape[1]), dtype=x.dtype)), dim=0)
    x = x.T
    print('pool x shape', x.shape)
    assert x.shape[1] % k == 0
    res = x.reshape(-1, k).max(dim=1)[0].reshape(x.shape[0], -1).T
    offsets = (res == 0) & (torch.cat((torch.zeros((1, res.shape[1]), dtype=res.dtype), res[: -1, :]), dim=0) == 2)
    res[offsets] = 1
    return res


class EMDATASET(Dataset):
    def __init__(self,
                 audio_path='NoteEM_audio',
                 labels_path='NoteEm_labels',
                 groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE,
                 instrument_map=None, update_instruments=False, transcriber=None,
                 conversion_map=None, shift_range=(-5, 6)):
        self.audio_path = audio_path
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.groups = groups
        self.conversion_map = conversion_map
        self.shift_range = shift_range
        self.file_list = self.files(self.groups)
        if instrument_map is None:
            self.get_instruments()
        else:
            self.instruments = instrument_map
            if update_instruments:
                self.add_instruments()
        self.transcriber = transcriber
        self.load_pts(self.file_list)
        self.data = []
        print('Reading files...')
        for input_files in tqdm(self.file_list):
            data = self.pts[input_files[0]]
            audio_len = len(data['audio'])
            minutes = audio_len // (SAMPLE_RATE * 60)
            copies = minutes
            # copies = 1
            for _ in range(copies):
                self.data.append(input_files)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def files(self, groups, skip_keyword=['I. Overture', 'BWV 1067 - 1. Ouverture', 'BWV 1068 - 1. Ouverture']):
        self.path = '/disk4/ben/PerformanceNet-master/Museopen16_flac'
        if not os.path.exists(self.path):
            self.path = 'Museopen16_flac'
        # self.path = '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen16AUG'

        tsvs_path = '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen_TSV_multi'
        if not os.path.exists(tsvs_path):
            tsvs_path = 'Museopen_TSV_multi'
        res = []
        ### Wind ensembles:
        good_ids = list(range(2075, 2084))

        # good_ids += list(range(1817, 1820))
        # good_ids += list(range(2202, 2205))
        # good_ids += list(range(2415, 2418))
        # good_ids += list(range(2504, 2508))

        # good_ids = list(range(1788, 1894))
        # good_ids += list(range(1727, 1731))
        # good_ids += list(range(1788, 1792))
        # good_ids += list(range(1792, 1808))
        # good_ids += list(range(1811, 1814))
        # good_ids += list(range(1817, 1820))
        # good_ids += list(range(1822, 1825))
        # good_ids += list(range(1828, 1830))
        # good_ids += list(range(1835, 1860))
        # good_ids += list(range(1872, 1894))


        for group in groups:
            try:
                tsvs = os.listdir(tsvs_path + '/' + group)
            except:
                tsvs = os.listdir(tsvs_path.split('/')[-1] + '/' + group)
            tsvs = sorted(tsvs)
            # for shft in range(-5, 6):
            for shft in range(self.shift_range[0], self.shift_range[1]):
                curr_fls_pth = self.path + '/' + group + '#{}'.format(shft)
                fls = os.listdir(curr_fls_pth)
                fls = [fl for fl in fls if fl.split('/')[-1].split('#')[0] not in ['2194', '2211', '2227', '2230', '2292', '2305', '2310']]
                fls = sorted(fls)
                cnt = 0
                for f, t in zip(fls, tsvs):
                    print('ft', f, t)
                    # if cnt > 3:
                    #     print('breaking at', cnt)
                    #     break
                    cnt += 1
                    # #### MusicNet
                    if 'MusicNet' in group:
                        if all([str(elem) not in f for elem in good_ids]):
                            continue
                    if skip_keyword is not None and any([elem in f for elem in skip_keyword]):
                        print('skipping', f)
                        continue
                    print('adding', f, t)
                    res.append((curr_fls_pth + '/' + f, tsvs_path + '/' + group + '/' + t))
        return res

    def get_instruments(self):
        instruments = set()
        for _, f in self.file_list:
            print('loading midi from', f)
            events = np.loadtxt(f, delimiter='\t', skiprows=1)
            curr_instruments = set(events[:, -1])
            instruments = instruments.union(curr_instruments)
        instruments = [int(elem) for elem in instruments if elem < 115]
        instruments = list(set(instruments))
        if 0 in instruments:
            piano_ind = instruments.index(0)
            instruments.pop(piano_ind)
            instruments.insert(0, 0)
        self.instruments = instruments
        self.instruments = list(set(self.instruments) - set(range(88, 104)) - set(range(112, 150)))
        print('Dataset instruments:', self.instruments)
        print('Total:', len(self.instruments), 'instruments')

    def add_instruments(self):
        for _, f in self.file_list:
            events = np.loadtxt(f, delimiter='\t', skiprows=1)
            curr_instruments = set(events[:, -1])
            new_instruments = curr_instruments - set(self.instruments)
            self.instruments += list(new_instruments)
        instruments = [int(elem) for elem in self.instruments if (elem < 115)]
        self.instruments = instruments

    def __getitem__(self, index):
        data = self.load(*self.data[index])
        result = dict(path=data['path'])
        midi_length = len(data['label'])
        n_steps = self.sequence_length // HOP_LENGTH
        step_begin = self.random.randint(midi_length - n_steps)
        step_end = step_begin + n_steps
        begin = step_begin * HOP_LENGTH
        end = begin + self.sequence_length
        result['audio'] = data['audio'][begin:end]
        diff = self.sequence_length - len(result['audio'])
        result['audio'] = torch.cat((result['audio'], torch.zeros(diff, dtype=result['audio'].dtype)))
        result['audio'] = result['audio'].to(self.device)
        result['label'] = data['label'][step_begin:step_end, ...]
        result['label'] = result['label'].to(self.device)
        if 'velocity' in data:
            result['velocity'] = data['velocity'][step_begin:step_end, ...].to(self.device)
            result['velocity'] = result['velocity'].float() / 128.

        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()

        if 'onset_mask' in data:
            result['onset_mask'] = data['onset_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_mask' in data:
            result['frame_mask'] = data['frame_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_pos_label' in data:
            result['frame_pos_label'] = data['frame_pos_label'][step_begin:step_end, ...].to(self.device).float()

        # print('get item label', result['label'].shape)
        shape = result['frame'].shape
        keys = N_KEYS
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        # frame and offset currently do not differentiate between instruments,
        # so we compress them across instrument and save a copy of the original,
        # as 'big_frame' and 'big_offset'
        result['big_frame'] = result['frame']
        result['frame'], _ = result['frame'].reshape(new_shape).max(axis=-2)
        result['big_offset'] = result['offset']
        result['offset'], _ = result['offset'].reshape(new_shape).max(axis=-2)

        return result

    def load(self, audio_path, tsv_path):
        data = self.pts[audio_path]
        if len(data['audio'].shape) > 1:
            data['audio'] = (data['audio'].float().mean(dim=-1)).short()
        if 'label' in data:
            return data
        else:
            piece, part = audio_path.split('/')[-2:]
            piece_split = piece.split('#')
            if len(piece_split) == 2:
                piece, shift1 = piece_split
            else:
                piece, shift1 = '#'.join(piece_split[:2]), piece_split[-1]
            part_split = part.split('#')
            if len(part_split) == 2:
                part, shift2 = part_split
            else:
                part, shift2 = '#'.join(part_split[:2]), part_split[-1]
            shift2, _ = shift2.split('.')
            assert shift1 == shift2
            shift = shift1
            assert shift != 0
            orig = audio_path.replace('#{}'.format(shift), '#0')
            res = {}
            res['label'] = shift_label(self.pts[orig]['label'], int(shift))
            res['path'] = audio_path
            res['audio'] = data['audio']
            if 'synth' in data:
                res['synth'] = data['synth']
            if 'velocity' in self.pts[orig]:
                res['velocity'] = shift_label(self.pts[orig]['velocity'], int(shift))
            if 'onset_mask' in self.pts[orig]:
                res['onset_mask'] = shift_label(self.pts[orig]['onset_mask'], int(shift))
            if 'frame_mask' in self.pts[orig]:
                res['frame_mask'] = shift_label(self.pts[orig]['frame_mask'], int(shift))
            if 'frame_pos_label' in self.pts[orig]:
                res['frame_pos_label'] = shift_label(self.pts[orig]['frame_pos_label'], int(shift))
            return res

    def load_pts(self, files):
        self.pts = {}
        print('loading pts...')
        for flac, tsv in tqdm(files):
            print('flac, tsv', flac, tsv)
            if os.path.isfile(self.labels_path + '/' +
                              flac.split('/')[-1].replace('.flac', '.pt')):
                self.pts[flac] = torch.load(self.labels_path + '/' +
                              flac.split('/')[-1].replace('.flac', '.pt'))
            else:
                if flac.count('#') != 2:
                    print('two #', flac)
                audio, sr = soundfile.read(flac, dtype='int16')
                if len(audio.shape) == 2:
                    audio = audio.astype(float).mean(axis=1)
                else:
                    audio = audio.astype(float)
                #### rms normalization
                # audio_rms = get_rms(audio)
                # print('audio rms', audio_rms)
                # audio = rms_normalize(audio)
                ####
                audio = audio.astype(np.int16)
                print('audio len', len(audio))
                assert sr == SAMPLE_RATE
                audio = torch.ShortTensor(audio)
                if '#0' not in flac:
                    assert '#' in flac
                    data = {'audio': audio}
                    self.pts[flac] = data
                    torch.save(data,
                               self.labels_path + '/' + flac.split('/')[-1]
                               .replace('.flac', '.pt').replace('.mp3', '.pt'))
                    continue
                midi = np.loadtxt(tsv, delimiter='\t', skiprows=1)
                unaligned_label = midi_to_frames(midi, self.instruments, conversion_map=self.conversion_map)
                data = dict(path=self.labels_path + '/' + flac.split('/')[-1],
                            audio=audio, unaligned_label=unaligned_label)
                torch.save(data, self.labels_path + '/' + flac.split('/')[-1]
                               .replace('.flac', '.pt').replace('.mp3', '.pt'))
                self.pts[flac] = data

    '''
    Update labels. 
    POS, NEG - pseudo labels positive and negative thresholds.
    PITCH_POS - pseudo labels positive thresholds for the pitch-only classes.
    first - is this the first labelling iteration.
    update - should the labels indeed be updated - if not, just saves the output.
    BEST_BON - if true, will update labels only if the bag of notes distance between the unaligned midi and the prediction improved.
    Bag of notes distance is computed based on pitch only.
    '''
    def update_pts(self, transcriber, POS=1.1, NEG=-0.001, FRAME_POS=0.5,
                   to_save=None, first=False, update=True, BEST_BON=False):
        print('Updating pts...')
        print('POS, NEG', POS, NEG)
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        print('there are', len(self.pts), 'pts')
        for flac, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            audio_inp = data['audio'].float() / 32768.
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
                print('long audio, splitting to {} segments'.format(n_segments))
                seg_len = audio_inp_len // n_segments
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                vel_preds = []
                for i_s in range(n_segments):
                    curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
                    curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                    vel_preds.append(curr_velocity_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
                velocity_pred = torch.cat(vel_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)
            print('done predicting.')
            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()
            offset_pred = offset_pred.detach().squeeze().cpu()


            peaks = get_peaks(onset_pred, 3) # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
            onset_pred[~peaks] = 0

            unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
            unaligned_frames = (data['unaligned_label'] >= 2).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()
            offset_pred_np = offset_pred.numpy()


            ####
            pred_bag_of_notes = (onset_pred_np[:, -N_KEYS:] >= 0.5).sum(axis=0)
            gt_bag_of_notes = unaligned_onsets[:, -N_KEYS:].astype(bool).sum(axis=0)

            # pred_bag_of_notes = max_inst(onset_pred_np, 0.5)[:, : -N_KEYS].sum(axis=0)
            # gt_bag_of_notes = unaligned_onsets[:, :-N_KEYS].astype(bool).sum(axis=0)



            bon_dist = (((pred_bag_of_notes - gt_bag_of_notes) ** 2).sum()) ** 0.5
            # print('pred bag of notes', pred_bag_of_notes)
            # print('gt bag of notes', gt_bag_of_notes)
            bon_dist /= gt_bag_of_notes.sum()
            print('bag of notes dist', bon_dist)
            ####

            # We align based on likelihoods regardless of the octave (chroma features)
            onset_pred_comp = compress_across_octave(onset_pred_np[:, -N_KEYS:])
            onset_label_comp = compress_across_octave(unaligned_onsets[:, -N_KEYS:])
            # We can do DTW on super-frames since anyway we search for local max afterwards
            onset_pred_comp = compress_time(onset_pred_comp, DTW_FACTOR)
            onset_label_comp = compress_time(onset_label_comp, DTW_FACTOR)
            print('dtw lengths', len(onset_pred_comp), len(onset_label_comp))
            init_time = time.time()
            alignment = dtw(onset_pred_comp, onset_label_comp, dist_method='euclidean',
                            )
            finish_time = time.time()
            print('DTW took {} seconds.'.format(finish_time - init_time))
            index1, index2 = alignment.index1, alignment.index2
            matches1, matches2 = get_matches(index1, index2), get_matches(index2, index1)

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_offsets = np.zeros(onset_pred_np.shape, dtype=bool)
            frame_pos_label = np.zeros(onset_pred_np.shape, dtype=float)
            vel_pred_np = velocity_pred.detach().squeeze().cpu().numpy()

            # We go over onsets (t, f) in the unaligned midi. For each onset, we find its approximate time based on DTW,
            # then find its precise time with likelihood local max
            t_win = [0, 0, 0, 0]
            for t, f in zip(*unaligned_onsets.nonzero()):
                t_comp = t // DTW_FACTOR
                t_src = matches2[t_comp]
                t_sources = list(range(DTW_FACTOR * min(t_src), DTW_FACTOR * max(t_src) + 1))
                # we extend the search area of local max to be ~0.5 second:
                t_sources_extended = get_margin(t_sources, len(aligned_onsets))
                # eliminate occupied positions. Allow only a single onset per 5 frames:
                existing_eliminated = [t_source for t_source in t_sources_extended if (aligned_onsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(existing_eliminated) > 0:
                    t_sources_extended = existing_eliminated

                t_src = max(t_sources_extended, key=lambda x: onset_pred_np[x, f]) # t_src is the most likely time in the local neighborhood for this note onset
                f_pitch = (len(self.instruments) * N_KEYS) + (f % N_KEYS)
                if onset_pred_np[t_src, f_pitch] < NEG: # filter negative according to pitch-only likelihood (can use f instead)
                    continue
                aligned_onsets[t_src, f] = 1 # set the label

                # Now we need to decide note duration and offset time. Find note length in unaligned midi:
                t_off = t
                while t_off < len(unaligned_frames) and unaligned_frames[t_off, f]:
                    t_off += 1
                note_len = t_off - t # this is the note length in the unaligned midi. We need note length in the audio.

                # option 1: use mapping, traverse note length in the unaligned midi, and then use the reverse mapping:
                try:
                    t_off_src1 = max(matches2[(DTW_FACTOR * max(matches1[t_src // DTW_FACTOR]) + note_len) // DTW_FACTOR]) * DTW_FACTOR
                    t_off_src1 = max(t_src + 1, t_off_src1)
                except Exception as e:
                    t_off_src1 = len(aligned_offsets)
                # option 2: use relative note length
                t_off_src2 = t_src + int(note_len * (len(aligned_onsets) / len(unaligned_onsets)))
                t_off_src2 = min(len(aligned_onsets), t_off_src2)

                # option 3: frame prediction
                t_off_src3 = t_src
                while t_off_src3 < len(frame_pred_np) and frame_pred_np[t_off_src3, f % N_KEYS] >= FRAME_POS:
                    t_off_src3 += 1


                t_off_longest = max([t_off_src1, t_off_src2, t_off_src3])  # we choose the longest
                t_off_shortest = min([t_off_src1, t_off_src2, t_off_src3])  # we choose the shortest
                offset_range = list(range(max(0, t_off_shortest), min(t_off_longest + 1, len(offset_pred_np))))
                offset_range_ex = get_margin(offset_range, len(offset_pred_np), WINDOW_SIZE_SRC=int(0.5 * SAMPLE_RATE / HOP_LENGTH), min_left=t_src + 3)
                offset_eliminated = [t_source for t_source in offset_range_ex if (aligned_offsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(offset_eliminated) > 0:
                    offset_range_ex = offset_eliminated
                t_off_most_likely = max(offset_range_ex, key=lambda x: offset_pred_np[x, f % N_KEYS])

                # t_off_most_likely = max(range(max(0, t_src + 3, t_off_shortest - 20), min(len(offset_pred), t_off_longest + 20)), key=lambda x: offset_pred_np[x, f % N_KEYS])

                winner = np.argmax([t_off_src1, t_off_src2, t_off_src3, t_off_most_likely])
                t_win[winner] += 1
                # off_range = t_off_longest - t_off_shortest
                # print('off range', off_range, off_range * HOP_LENGTH / SAMPLE_RATE)
                # print('times', t_off_src1 - t_src, t_off_src2 - t_src, t_off_src3 - t_src, t_off_most_likely - t_src)
                # t_off_src = t_off_longest
                # t_off_src = t_off_shortest
                t_off_src = t_off_most_likely
                aligned_frames[t_src: t_off_src, f] = 1

                if t_off_src < len(aligned_offsets):
                    aligned_offsets[t_off_src, f] = 1
            print('twin', t_win)
            # eliminate instruments that do not exist in the unaligned midi
            inactive_instruments, active_instruments_list = get_inactive_instruments(unaligned_onsets, len(aligned_onsets))
            onset_pred_np[inactive_instruments] = 0

            # pseudo_onsets = (onset_pred_np >= POS) & (~aligned_onsets)
            print('pseudo max inst')
            pseudo_onsets = max_inst(onset_pred_np, POS) & (~aligned_onsets)
            pseudo_onsets_05 = max_inst(onset_pred_np, 0.5) & (~aligned_onsets)

            print('done max inst')
            inst_only = len(self.instruments) * N_KEYS
            if first: # do not use pseudo labels for instruments in first labelling iteration since the model doesn't distinguish yet
                pseudo_onsets[:, : inst_only] = 0
                pseudo_onsets_05[:, : inst_only] = 0
            else:
                print('not first')
            onset_label = np.maximum(pseudo_onsets, aligned_onsets)
            onset_label_05 = np.maximum(pseudo_onsets_05, aligned_onsets)


            onsets_unknown = (onset_pred_np >= 0.5) & (~onset_label) # for mask
            if first: # do not use mask for instruments in first labelling iteration since the model doesn't distinguish yet between instruments
                onsets_unknown[:, : inst_only] = 0
            onset_mask = torch.from_numpy(~onsets_unknown).byte()
            # onset_mask = torch.ones(onset_label.shape).byte()

            pseudo_frames = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            pseudo_offsets = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            for t, f in zip(*onset_label.nonzero()):
                t_off = t
                while t_off < len(pseudo_frames) and frame_pred[t_off, f % N_KEYS] >= FRAME_POS:
                    t_off += 1
                pseudo_frames[t: t_off, f] = 1
                if t_off < len(pseudo_offsets):
                    pseudo_offsets[t_off, f] = 1
            # frame_label = np.maximum(pseudo_frames, aligned_frames)
            frame_label = aligned_frames
            # offset_label = get_diff(frame_label, offset=True)
            offset_label = aligned_offsets

            frames_pitch_only = frame_label[:, -N_KEYS:]
            frames_unknown = (frame_pred_np >= 0.5) & (~frames_pitch_only)
            frame_mask = torch.from_numpy(~frames_unknown).byte()
            # frame_mask = torch.ones(frame_pred.shape).byte()

            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            activation_label = np.maximum(onset_label, frame_label)
            new_vels = np.zeros(vel_pred_np.shape, dtype=vel_pred_np.dtype)
            pos_f = lambda x: x #** 2
            for t, f in zip(*onset_label.nonzero()):
                t_end = t
                while t_end < len(frame_label) and activation_label[t_end, f]:
                    t_end += 1
                num_pts = t_end - t if t_end - t_end >= 0 else 1
                curr_pos_encoding = np.linspace(1., 0., num_pts)
                if t_end - t < 0:
                    print('pos short note')
                frame_pos_label[t: t_end, f] = pos_f(curr_pos_encoding)
                note_estimated_vel = vel_pred_np[t, f]
                new_vels[t: t_end, f] = note_estimated_vel

            #### Statistics:
            d_insts = {}
            for i_ins, ins in enumerate(self.instruments):
                curr_ins_onsets = aligned_onsets[:, i_ins * N_KEYS: (i_ins + 1) * N_KEYS]
                curr_ins_vels = vel_pred_np[:, i_ins * N_KEYS: (i_ins + 1) * N_KEYS]
                # Mean vel per instrument:
                curr_ins_mean_vel = (curr_ins_onsets * curr_ins_vels).sum() / curr_ins_onsets.sum()
                print(i_ins, ins, 'mean vel:', curr_ins_mean_vel)
                d_insts[i_ins, ins, 'mean_vel'] = curr_ins_mean_vel
                # Active notes per instrument:
                note_sums = curr_ins_onsets.sum(axis=0).astype(int)
                shifted_note_sums = np.zeros(note_sums.shape, dtype=int)
                for shft in range(-5, 6):
                    padding = (0, -shft) if shft < 0 else (shft, 0)
                    curr_shifted = np.pad(note_sums, padding)
                    curr_shifted = curr_shifted[-shft:] if shft <= 0 else curr_shifted[: -shft]
                    shifted_note_sums += curr_shifted
                print('note appearances:', shifted_note_sums)
                shited_notes_sums = torch.from_numpy(shifted_note_sums)
                d_insts[i_ins, ins, 'notes'] = shited_notes_sums

            ####

            if to_save is not None:
                time_now = datetime.now().strftime('%y%m%d-%H%M%S')
                curr_logdir = '/'.join(to_save.split('/')[: -1])
                torch.save(d_insts, curr_logdir + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_stats.pt')
                torch.save(d_insts, to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_stats_' + time_now + '.pt')
                save_midi_alignments_and_predictions(to_save, data['path'], self.instruments,
                                                     onset_label_05,
                                         aligned_onsets, aligned_frames,
                                         onset_pred_np, frame_pred_np, prefix='')
            if update:
                if not BEST_BON or bon_dist < data.get('BON', float('inf')):
                    data['label'] = torch.from_numpy(label).byte()
                    data['onset_mask'] = onset_mask
                    data['frame_mask'] = frame_mask
                    data['frame_pos_label'] = torch.from_numpy(frame_pos_label).float()

                    velocity_pred = velocity_pred.detach().squeeze().cpu()
                    velocity_pred = (128. * velocity_pred)
                    velocity_pred[velocity_pred < 0.] = 0.
                    velocity_pred[velocity_pred > 127.] = 127.
                    velocity_pred = velocity_pred.byte()
                    data['velocity'] = velocity_pred

                if bon_dist < data.get('BON', float('inf')):
                    print('Bag of notes distance improved from {} to {}'.format(data.get('BON', float('inf')), bon_dist))
                    data['BON'] = bon_dist

                    if to_save is not None:
                        os.makedirs(to_save + '/BEST', exist_ok=True)
                        save_midi_alignments_and_predictions(to_save + '/BEST', data['path'], self.instruments,
                                                             onset_label,
                                                             aligned_onsets, aligned_frames,
                                                             onset_pred_np, frame_pred_np, prefix='BEST', use_time=False)



            del audio_inp
            try:
                del mel
            except:
                pass
            del onset_pred
            del offset_pred
            del frame_pred
            del velocity_pred
            torch.cuda.empty_cache()

    '''
        Update labels. Use only alignment without pseudo-labels.
    '''

    def update_pts_vanilla(self, transcriber,
                   to_save=None, first=False, update=True):
        print('Updating pts...')
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        print('there are', len(self.pts), 'pts')
        for flac, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            audio_inp = data['audio'].float() / 32768.
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
                print('long audio, splitting to {} segments'.format(n_segments))
                seg_len = audio_inp_len // n_segments
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                vel_preds = []
                for i_s in range(n_segments):
                    curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
                    curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                    vel_preds.append(curr_velocity_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
                velocity_pred = torch.cat(vel_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)
            print('done predicting.')
            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()

            peaks = get_peaks(onset_pred, 3)  # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
            onset_pred[~peaks] = 0

            unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
            unaligned_frames = (data['unaligned_label'] >= 2).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()

            # We align based on likelihoods regardless of the octave (chroma features)
            onset_pred_comp = compress_across_octave(onset_pred_np[:, -N_KEYS:])
            onset_label_comp = compress_across_octave(unaligned_onsets[:, -N_KEYS:])
            # We can do DTW on super-frames since anyway we search for local max afterwards
            onset_pred_comp = compress_time(onset_pred_comp, DTW_FACTOR)
            onset_label_comp = compress_time(onset_label_comp, DTW_FACTOR)
            print('dtw lengths', len(onset_pred_comp), len(onset_label_comp))
            init_time = time.time()
            alignment = dtw(onset_pred_comp, onset_label_comp, dist_method='euclidean',
                            )
            finish_time = time.time()
            print('DTW took {} seconds.'.format(finish_time - init_time))
            index1, index2 = alignment.index1, alignment.index2
            matches1, matches2 = get_matches(index1, index2), get_matches(index2, index1)

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_offsets = np.zeros(onset_pred_np.shape, dtype=bool)

            # We go over onsets (t, f) in the unaligned midi. For each onset, we find its approximate time based on DTW,
            # then find its precise time with likelihood local max
            for t, f in zip(*unaligned_onsets.nonzero()):
                t_comp = t // DTW_FACTOR
                t_src = matches2[t_comp]
                t_sources = list(range(DTW_FACTOR * min(t_src), DTW_FACTOR * max(t_src) + 1))
                # we extend the search area of local max to be ~0.5 second:
                t_sources_extended = get_margin(t_sources, len(aligned_onsets))
                # eliminate occupied positions. Allow only a single onset per 5 frames:
                existing_eliminated = [t_source for t_source in t_sources_extended if (aligned_onsets[t_source - 2: t_source + 3, f] == 0).all()]
                if len(existing_eliminated) > 0:
                    t_sources_extended = existing_eliminated

                t_src = max(t_sources_extended, key=lambda x: onset_pred_np[x, f])  # t_src is the most likely time in the local neighborhood for this note onset
                f_pitch = (len(self.instruments) * N_KEYS) + (f % N_KEYS)
                aligned_onsets[t_src, f] = 1  # set the label
                # Now we need to decide note duration and offset time. Find note length in unaligned midi:
                t_off = t
                while t_off < len(unaligned_frames) and unaligned_frames[t_off, f]:
                    t_off += 1
                note_len = t_off - t  # this is the note length in the unaligned midi. We need note length in the audio.

                # option 1: use mapping, traverse note length in the unaligned midi, and then use the reverse mapping:
                try:
                    t_off_src1 = max(matches2[(DTW_FACTOR * max(matches1[t_src // DTW_FACTOR]) + note_len) // DTW_FACTOR]) * DTW_FACTOR
                    t_off_src1 = max(t_src + 1, t_off_src1)
                except Exception as e:
                    t_off_src1 = len(aligned_offsets)
                # option 2: use relative note length
                t_off_src2 = t_src + int(note_len * (len(aligned_onsets) / len(unaligned_onsets)))
                t_off_src2 = min(len(aligned_onsets), t_off_src2)

                t_off_src = t_off_src2  # we choose option 2
                aligned_frames[t_src: t_off_src, f] = 1

                if t_off_src < len(aligned_offsets):
                    aligned_offsets[t_off_src, f] = 1

            # eliminate instruments that do not exist in the unaligned midi
            inactive_instruments, active_instruments_list = get_inactive_instruments(unaligned_onsets, len(aligned_onsets))
            onset_pred_np[inactive_instruments] = 0

            onset_label = aligned_onsets
            frame_label = aligned_frames
            offset_label = aligned_offsets
            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            if to_save is not None:
                inst_only = len(self.instruments) * N_KEYS
                time_now = datetime.now().strftime('%y%m%d-%H%M%S')
                frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_' + time_now + '.mid',
                            aligned_onsets[:, : inst_only], aligned_frames[:, : inst_only],
                            64. * aligned_onsets[:, : inst_only],
                            inst_mapping=self.instruments)
                frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_alignment_pitch_' + time_now + '.mid',
                                  aligned_onsets[:, -N_KEYS:], aligned_frames[:, -N_KEYS:],
                                  64. * aligned_onsets[:, -N_KEYS:])
                predicted_onsets = onset_pred_np >= 0.5
                predicted_frames = frame_pred_np >= 0.5
                frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_' + time_now + '.mid',
                            predicted_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                            64. * predicted_onsets[:, : inst_only],
                            inst_mapping=self.instruments)
                frames2midi_pitch(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_pitch_' + time_now + '.mid',
                                  predicted_onsets[:, -N_KEYS:], predicted_frames[:, -N_KEYS:],
                                  64. * predicted_onsets[:, -N_KEYS:])
                if len(self.instruments) > 1:
                    max_pred_onsets = max_inst(onset_pred_np)
                    frames2midi(to_save + '/' + data['path'].replace('.flac', '').split('/')[-1] + '_pred_max_' + time_now + '.mid',
                                max_pred_onsets[:, : inst_only], predicted_frames[:, : inst_only],
                                64. * max_pred_onsets[:, : inst_only],
                                inst_mapping=self.instruments)
            if update:
                data['label'] = torch.from_numpy(label).byte()

            velocity_pred = velocity_pred.detach().squeeze().cpu()
            # velocity_pred = torch.from_numpy(new_vels)
            velocity_pred = (128. * velocity_pred)
            velocity_pred[velocity_pred < 0.] = 0.
            velocity_pred[velocity_pred > 127.] = 127.
            velocity_pred = velocity_pred.byte()
            if update:
                data['velocity'] = velocity_pred

            del audio_inp
            try:
                del mel
            except:
                pass
            del onset_pred
            del offset_pred
            del frame_pred
            del velocity_pred
            torch.cuda.empty_cache()

def powerset(s):
    if len(s) == 0:
        return [[]]
    without_last = powerset(s[: -1])
    last = s[-1]
    return without_last + [elem + [last] for elem in without_last]

class AudioDataset(Dataset):
    def __init__(self, pth, device='cuda', sequence_length=SEQ_LEN, use_fma=True, use_fma_large=True, file_groups=5, train_groups=None,
                 pitch_shift=True, shift_range=None):
        super(AudioDataset, self).__init__()
        self.device = device
        self.sequence_length = sequence_length
        self.files = list(glob(pth + '/**/*.flac', recursive=True))
        self.file_groups = file_groups
        self.train_groups = train_groups
        print('init groups', self.train_groups)

        print('file groups', self.file_groups)

        print('files before', len(self.files))
        self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Beethoven Piano Trios*/**/*.flac', recursive=True))
        self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Beethoven Violin Sonatas*/**/*.flac', recursive=True))
        self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Bach Sonatas for Violin and Harpsichord B*/**/*.flac', recursive=True))
        self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Bach Trio Sonatas*/**/*.flac', recursive=True))

        print('files after adding', len(self.files))
        if use_fma and not use_fma_large:
            print('using fma_medium')
            self.files += list(glob('/disk3/ben/fma_medium/**/*.flac', recursive=True))
        elif use_fma and use_fma_large:
            print('using fma_large')
            self.files += list(glob('/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/**/*.flac', recursive=True))

        # print('files after fma')

        random.shuffle(self.files)
        # self.files = self.files[: 500]
        # print('files', self.files)

        self.load_pts(self.files, pitch_shift=pitch_shift, shift_range=shift_range)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, item):
        data = self.pts[item]
        result = dict(path=data['path'])
        audio_len = len(data['audio'])
        if audio_len - self.sequence_length > 0:
            begin = np.random.randint(audio_len - self.sequence_length)
            end = begin + self.sequence_length
            result['audio'] = data['audio'][begin:end]
        else:
            print('audio too short, len', audio_len, 'padding')
            result['audio'] = torch.nn.functional.pad(data['audio'], (0, self.sequence_length - audio_len))
        result['audio'] = result['audio'].to(self.device)
        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)
        return result

    def load_pts(self, file_list, pitch_shift=True, shift_range=None):
        print('loading pts...')
        file_list_new = []
        file_list_fma = []
        disk4_cnt = 0
        fma_cnt = 0
        fma_large_cnt = 0
        disk3_cnt = 0
        for f in tqdm(file_list):
            if 'Maestro' in f or 'GuitarSet' in f or 'Schubert_Sonata' in f or 'Debussy' in f:
                continue
            # if all(['#{}'.format(i) not in f for i in [-2, -1, 0, 1, 2, 3, 4]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
            # if all(['#{}'.format(i) not in f for i in [-1, 0, 1, 2]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
            # if all(['#{}'.format(i) not in f for i in [0]]):
            if all(['#{}'.format(i) not in f for i in [-2, -1, 0, 1, 2, 3]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
                continue
            if '/disk4/ben/PerformanceNet-master/Museopen16_flac' in f:
                disk4_cnt += 1
            if '/disk3/ben/fma_medium' in f:
                fma_cnt += 1
            if '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' in f:
                fma_large_cnt += 1
            if '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen16AUG' in f:
                disk3_cnt += 1
            (file_list_fma if 'fma_large/fma_large_flac' in f or '/disk3/ben/fma_medium' in f else file_list_new).append(f)
        print('cnts disk3, disk4, fma, fma_large', disk3_cnt, disk4_cnt, fma_cnt, fma_large_cnt)

        self.file_list = file_list_new
        self.file_list_fma = file_list_fma
        print('file lists lengths new and fma', len(self.file_list), len(self.file_list_fma))
        random.shuffle(self.file_list)
        random.shuffle(self.file_list_fma)
        # self.len_group = len(self.file_list) // self.file_groups
        self.len_group_fma = len(self.file_list_fma) // self.file_groups

        self.curr_group = -1
        self.pts = None
        self.update_pts(initial=True, pitch_shift=pitch_shift, shift_range=shift_range)


    def update_pts(self, initial=False, pitch_shift=True, shift_range=None):
        print('update pts, self train_groups', self.train_groups, 'shift range', shift_range)
        maybe_mean = lambda audio: audio.astype(float).mean(axis=1).astype(np.int16) if len(audio.shape) == 2 else audio

        if self.train_groups:
            self.pts = []
            for g in self.train_groups:
                print('extending', g, 'before', len(self.pts))
                if pitch_shift:
                    assert shift_range is not None
                    curr_pths = []
                    for shft_r in shift_range:
                        print('extending for shift', shft_r)
                        curr_shift_pths = glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/{}*#{}/**/*.flac'.format(g, shft_r), recursive=True)
                        curr_pths += list(curr_shift_pths)
                else:
                    print('not using pitch shift')
                    curr_pths = glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/{}*#0/**/*.flac'.format(g), recursive=True)
                if 'MusicNet' in g:
                    # if 'MusicNetTest' in g:
                    #     for cp in curr_pths:
                    #         if '#0' in cp:
                    #             print('cp', cp)
                    #     print('musicnet test', curr_pths)
                    good_ids = list(range(2075, 2084))
                    curr_pths = [el for el in curr_pths if any([str(sel) in el for sel in good_ids])]
                    # print('musicnet after', curr_pths)
                self.pts.extend([{'path': el, 'audio': torch.ShortTensor(maybe_mean(soundfile.read(el, dtype='int16')[0]))} for i_el, el in enumerate(tqdm(curr_pths))])
                print('extended after', len(self.pts))

        # print('updating pts audio daataset 1')
        # maybe_mean = lambda audio: audio.astype(float).mean(axis=1).astype(np.int16) if len(audio.shape) == 2 else audio
        # self.pts = [{'path': el, 'audio': torch.ShortTensor(maybe_mean(soundfile.read(el, dtype='int16')[0]))} for el in tqdm(glob('/disk4/ben/PerformanceNet-master/MelTestAudio/**/*.flac', recursive=True))]
        #
        # self.pts = [{'path': el, 'audio': torch.ShortTensor(maybe_mean(soundfile.read(el, dtype='int16')[0]))} for el in tqdm(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Rousset Goldberg#*/**/*.flac', recursive=True))]
        # self.pts = []
        # print('self len pts 1', len(self.pts))
        # print('extending...')
        # self.pts.extend([{'path': el, 'audio': torch.ShortTensor(maybe_mean(soundfile.read(el, dtype='int16')[0]))} for el in tqdm(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Beethoven Symphony*/**/*.flac', recursive=True))])
        # print('self len pts 2', len(self.pts))
        return

        if initial:
            file_list = self.file_list
            print('initial load pts...')
            res = []
            for f in tqdm(file_list):
                audio, sr = soundfile.read(f, dtype='int16')
                assert sr == SAMPLE_RATE
                # assert audio.max() <= 1. and audio.min() >= -1.
                if len(audio.shape) == 2:
                    audio = audio.astype(float).mean(axis=1).astype(np.int16)
                    # print('stereo to mono', f)
                if len(audio) < self.sequence_length + 10:
                    print('skipping', f)
                    continue
                audio = torch.ShortTensor(audio)
                seconds = len(audio) // SAMPLE_RATE
                copies = max(1, seconds // 30)
                # print('copies', f, copies)
                for _ in range(copies):
                    res.append({'path': f, 'audio': audio})
            random.shuffle(res)
            self.pts = res
            print('done')

        else:
            new_pts = []
            print('cleaning fma pts. Old len:', len(self.pts))
            removed_count = 0
            for pt in tqdm(self.pts):
                if 'fma_large/fma_large_flac' not in pt['path'] and '/disk3/ben/fma_medium' not in pt['path']:
                    new_pts.append(pt)
                else:
                    removed_count += 1
            print('removed', removed_count)
            del self.pts
            print('done')
            self.pts = new_pts
            print('new len', len(self.pts))


        file_list = random.sample(self.file_list_fma, self.len_group_fma)
        print('chose files from fma', len(file_list), 'out of', len(self.file_list_fma))
        print('loading fma pts...')
        skipped_cnt = 0
        # res = []
        for f in tqdm(file_list):
            audio, sr = soundfile.read(f, dtype='int16')
            assert sr == SAMPLE_RATE
            # assert audio.max() <= 1. and audio.min() >= -1.
            if len(audio.shape) == 2:
                audio = audio.astype(float).mean(axis=1).astype(np.int16)
                # print('stereo to mono', f)
            if len(audio) < self.sequence_length + 10:
                print('skipping', f)
                skipped_cnt += 1
                continue
            audio = torch.ShortTensor(audio)
            self.pts.append({'path': f, 'audio': audio})
        print('fma skipped cnt:', skipped_cnt)
        # self.pts.extend(res)

        print('data count:')
        data_fma = 0
        data_other = 0
        for pt in self.pts:
            if 'fma_large/fma_large_flac' not in pt['path'] and '/disk3/ben/fma_medium' not in pt['path']:
                data_other += 1
            else:
                data_fma += 1
        print('data other:', data_other, 'data fma:', data_fma)

    def get_ids(self, different_id_performances=None):
        # ids_in = list(self.pts.keys())

        ids_in = [el['path'] for el in self.pts]


        print('ids in', ids_in)
        # ids_out = [pth.split('/')[-2].split('#')[0] for pth in ids_in]
        # for i in range(len(ids_out)):
        #     if 'MusicNet' in ids_out[i]:
        #         ids_out[i] = 'MusicNet'
        # self.ids = list(set(ids_out))
        self.ids_map = {pth.split('/')[-1]: pth.split('/')[-2].split('#')[0] for pth in ids_in}
        for k, v in self.ids_map.items():
            # print('k, v', k, v)
            if 'MusicNet' in v:
                self.ids_map[k] = 'MusicNet'
            elif '1 Bach - Flute sonata in B minor BWV 1030 - Root and Van Delft' in v:
                self.ids_map[k] = 'Flute Sonata 1030 A'
                print('flute a')
            elif '2 BWV 1030 - Flute Sonata in B Minor' in v:
                self.ids_map[k] = 'Flute Sonata 1030 B'
                print('flute b')
            elif different_id_performances is not None and any([elem in v for elem in different_id_performances]):
                self.ids_map[k] = k.split('#')[0]
                print('id identity mapping', k, self.ids_map[k])
            else:
                print('mapping unchanged', k, v)
        self.ids = list(set(self.ids_map.values()))


    def get_ids_2(self, different_id_performances=None):
        # ids_in = list(self.pts.keys())

        ids_in = [el['path'] for el in self.pts]


        print('ids in', ids_in)
        # ids_out = [pth.split('/')[-2].split('#')[0] for pth in ids_in]
        # for i in range(len(ids_out)):
        #     if 'MusicNet' in ids_out[i]:
        #         ids_out[i] = 'MusicNet'
        # self.ids = list(set(ids_out))
        self.ids_map = {pth.split('/')[-1]: pth.split('/')[-2] for pth in ids_in}
        print('get ids 2')
        for k, v in self.ids_map.items():
            print('items k, v', k, v)
            shift = v.split('#')[1]
            assert int(shift) in list(range(-5, 6))
            if 'MusicNet' in v:
                self.ids_map[k] = 'MusicNet#{}'.format(shift)
            elif '1 Bach - Flute sonata in B minor BWV 1030 - Root and Van Delft' in v:
                self.ids_map[k] = 'Flute Sonata 1030 A#{}'.format(shift)
                print('flute a')
            elif '2 BWV 1030 - Flute Sonata in B Minor' in v:
                self.ids_map[k] = 'Flute Sonata 1030 B#{}'.format(shift)
                print('flute b')
            elif different_id_performances is not None and any([elem in v for elem in different_id_performances]):
                # self.ids_map[k] = k.split('#')[0]
                self.ids_map[k] = k.split('#')[0] + '#{}'.format(shift)
                print('id identity mapping', k, self.ids_map[k])
            else:
                print('mapping unchanged', k, v)
        self.ids = list(set(self.ids_map.values()))
        print('self ids', self.ids)


    def map_id(self, pth):
        pth_id = self.ids_map[pth.split('/')[-1]]
        res = self.ids.index(pth_id)
        # print('map id', pth, 'to', res)
        return res

class AudioDataset2(Dataset):
    def __init__(self, pth, device='cuda', sequence_length=SEQ_LEN, use_fma=True, use_fma_large=True, file_groups=5):
        super(AudioDataset2, self).__init__()
        self.device = device
        self.sequence_length = sequence_length
        self.files = list(glob(pth + '/**/*.flac', recursive=True))
        self.file_groups = file_groups
        print('file groups', self.file_groups)

        print('files before', len(self.files))
        if os.path.exists('/disk4/ben/PerformanceNet-master'):
            self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Beethoven Piano Trios*/**/*.flac', recursive=True))
            self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Beethoven Violin Sonatas*/**/*.flac', recursive=True))
            self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Bach Sonatas for Violin and Harpsichord B*/**/*.flac', recursive=True))
            self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/Bach Trio Sonatas*/**/*.flac', recursive=True))

            self.files += list(glob('/disk4/ben/PerformanceNet-master/Museopen16_flac/**/*.flac', recursive=True))


            print('files after adding', len(self.files))
            if use_fma and not use_fma_large:
                print('using fma_medium')
                self.files += list(glob('/disk3/ben/fma_medium/**/*.flac', recursive=True))
            elif use_fma and use_fma_large:
                print('using fma_large')
                self.files += list(glob('/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/**/*.flac', recursive=True))
        else:
            print('running on cluster')
            self.files += list(glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/Beethoven Piano Trios*/**/*.flac', recursive=True))
            self.files += list(glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/Beethoven Violin Sonatas*/**/*.flac', recursive=True))
            self.files += list(glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/Bach Sonatas for Violin and Harpsichord B*/**/*.flac', recursive=True))
            self.files += list(glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/Bach Trio Sonatas*/**/*.flac', recursive=True))

            self.files += list(glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/**/*.flac', recursive=True))

            print('files after adding', len(self.files))
            if use_fma and not use_fma_large:
                print('using fma_medium')
                self.files += list(glob('/home/dcor/benmaman/fma_medium/**/*.flac', recursive=True))
            elif use_fma and use_fma_large:
                print('using fma_large')
                self.files += list(glob('/home/dcor/benmaman/PerformanceNet-master/fma_large/fma_large_flac/**/*.flac', recursive=True))

        # print('files after fma')

        random.shuffle(self.files)
        # self.files = self.files[: 2000]
        # print('files', self.files)

        self.load_pts(self.files)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, item):
        data = self.pts[item]
        result = dict(path=data['path'])
        audio, sr = soundfile.read(data['path'], dtype='int16')
        assert sr == SAMPLE_RATE
        # assert audio.max() <= 1. and audio.min() >= -1.
        if len(audio.shape) == 2:
            audio = audio.astype(float).mean(axis=1).astype(np.int16)
            # print('stereo to mono', f)
        audio = torch.ShortTensor(audio)

        audio_len = len(audio)
        if audio_len - self.sequence_length > 0:
            begin = np.random.randint(audio_len - self.sequence_length)
            end = begin + self.sequence_length
            result['audio'] = audio[begin:end]
        else:
            print('audio too short, len', audio_len, 'padding')
            result['audio'] = torch.nn.functional.pad(audio, (0, self.sequence_length - audio_len))
        result['audio'] = result['audio'].to(self.device)
        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)
        return result

    def load_pts(self, file_list):
        # self.pts = [{'path': el, 'audio': } for el in glob('/disk4/ben/PerformanceNet-master/MelTestAudio/**/*.flac', recursive=True)]
        # return
        print('loading pts...')
        file_list_new = []
        file_list_fma = []
        disk4_cnt = 0
        fma_cnt = 0
        fma_large_cnt = 0
        disk3_cnt = 0
        for f in tqdm(file_list):
            if 'Maestro' in f or 'GuitarSet' in f or 'Schubert_Sonata' in f or 'Debussy' in f:
                continue
            # if all(['#{}'.format(i) not in f for i in [-2, -1, 0, 1, 2, 3, 4]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
            # if all(['#{}'.format(i) not in f for i in [-1, 0, 1, 2]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
            # if all(['#{}'.format(i) not in f for i in [0]]):
            # if all(['#{}'.format(i) not in f for i in [-2, -1, 0, 1, 2, 3]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
            if all(['#{}'.format(i) not in f for i in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]] + ['/disk3/ben/fma_medium' not in f, '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' not in f]):
                continue
            if '/disk4/ben/PerformanceNet-master/Museopen16_flac' in f:
                disk4_cnt += 1
            if '/disk3/ben/fma_medium' in f:
                fma_cnt += 1
            if '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac' in f:
                fma_large_cnt += 1
            if '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen16AUG' in f:
                disk3_cnt += 1
            (file_list_fma if 'fma_large/fma_large_flac' in f or '/disk3/ben/fma_medium' in f else file_list_new).append(f)
        print('cnts disk3, disk4, fma, fma_large', disk3_cnt, disk4_cnt, fma_cnt, fma_large_cnt)

        self.file_list = file_list_new
        self.file_list_fma = file_list_fma
        print('file lists lengths new and fma', len(self.file_list), len(self.file_list_fma))
        random.shuffle(self.file_list)
        random.shuffle(self.file_list_fma)
        # self.len_group = len(self.file_list) // self.file_groups
        self.len_group_fma = len(self.file_list_fma) // self.file_groups

        self.curr_group = -1
        self.pts = None
        self.update_pts(initial=True)


    def update_pts(self, initial=False):
        # self.pts = [{'path': el} for el in glob('/disk4/ben/PerformanceNet-master/MelTestAudio/**/*.flac', recursive=True)]
        # return
        print('updating pts')
        to_skip = ['/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/106/106586.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/140/140853.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/133/133735.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/138/138485.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113025.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/128/128538.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/131/131814.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/136/136871.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095269.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113023.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113019.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095287.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/098/098446.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113018.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095296.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/140/140854.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/114/114497.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/139/139671.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/126/126320.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095266.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/098/098567.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/126/126319.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095264.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/094/094784.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095257.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095260.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/087/087744.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113026.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/126/126318.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/129/129836.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113024.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095268.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113017.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/106/106585.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095252.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095270.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/131/131817.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095262.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/136/136874.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/087/087435.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/106/106589.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/071/071826.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/098/098451.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095256.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/102/102144.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/108/108855.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/138/138488.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095259.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/129/129840.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113028.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/139/139674.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/140/140791.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113022.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095263.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/097/097127.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/136/136869.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095288.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/126/126317.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113021.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113016.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113020.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/102/102139.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/080/080237.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095267.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/138/138484.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/106/106587.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/136/136868.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095289.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/131/131812.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095297.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095294.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/023/023431.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/126/126316.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095258.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/106/106588.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/113/113027.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/136/136870.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095261.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/140/140794.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095293.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/136/136876.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/131/131813.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/140/140790.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/095/095298.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/126/126321.flac', '/disk4/ben/PerformanceNet-master/fma_large/fma_large_flac/140/140789.flac']
        if initial:
            file_list = self.file_list
            print('initial load pts...')
            res = []
            ## option 1
            for f in tqdm(file_list):
                audio, sr = soundfile.read(f, dtype='int16')
                assert sr == SAMPLE_RATE
                # assert audio.max() <= 1. and audio.min() >= -1.
                if len(audio.shape) == 2:
                    audio = audio.astype(float).mean(axis=1).astype(np.int16)
                    # print('stereo to mono', f)
                if len(audio) < self.sequence_length + 10:
                    print('skipping', f)
                    continue
                ### reading from disk
                # audio = torch.ShortTensor(audio)
                ###
                seconds = len(audio) // SAMPLE_RATE
                copies = max(1, seconds // 30)
                # print('copies', f, copies)
                for _ in range(copies):
                    # res.append({'path': f, 'audio': audio}) # reading from disk
                    res.append({'path': f})
                del audio # reading from disk
            #### end option 1

            # #### option 2
            # print('res before')
            # res = [{'path': f} for f in file_list if f not in to_skip]
            # print('res after')
            # ### end option 2
            random.shuffle(res)
            self.pts = res
            print('done')

        else:
            new_pts = []
            print('cleaning fma pts. Old len:', len(self.pts))
            removed_count = 0
            for pt in tqdm(self.pts):
                if 'fma_large/fma_large_flac' not in pt['path'] and '/disk3/ben/fma_medium' not in pt['path']:
                    new_pts.append(pt)
                else:
                    removed_count += 1
            print('removed', removed_count)
            del self.pts
            print('done')
            self.pts = new_pts
            print('new len', len(self.pts))


        if self.len_group_fma < len(self.file_list_fma):
            print('fma subsampling')
            file_list = random.sample(self.file_list_fma, self.len_group_fma)
        else:
            file_list = self.file_list_fma[:]
            print('fma full')

        print('loading fma pts...')
        # res = []
        # #### option 1
        # skipped_cnt = 0
        # for f in tqdm(file_list):
        #     audio, sr = soundfile.read(f, dtype='int16')
        #     assert sr == SAMPLE_RATE
        #     # assert audio.max() <= 1. and audio.min() >= -1.
        #     if len(audio.shape) == 2:
        #         audio = audio.astype(float).mean(axis=1).astype(np.int16)
        #         # print('stereo to mono', f)
        #     if len(audio) < self.sequence_length + 10:
        #         print('skipping', f)
        #         skipped_cnt += 1
        #         del audio
        #         continue
        #     # audio = torch.ShortTensor(audio) # reading from disk
        #     # self.pts.append({'path': f, 'audio': audio})
        #     self.pts.append({'path': f}) #reading from disk
        #     del audio
        # print('fma skipped cnt:', skipped_cnt)
        # #### end option 1
        print('fma res before')
        fma_res = [{'path': f} for f in file_list if f not in to_skip]
        print('fma res after file list, to skip, fma_res', len(file_list), len(to_skip), len(fma_res))
        self.pts.extend(fma_res)


        # self.pts.extend(res)

        print('data count:')
        data_fma = 0
        data_other = 0
        for pt in self.pts:
            if 'fma_large/fma_large_flac' not in pt['path'] and '/disk3/ben/fma_medium' not in pt['path']:
                data_other += 1
            else:
                data_fma += 1
        print('data other:', data_other, 'data fma:', data_fma)


    def get_ids(self, different_id_performances=None):
        ids_in = list(self.pts.keys())
        print('ids in', ids_in)
        # ids_out = [pth.split('/')[-2].split('#')[0] for pth in ids_in]
        # for i in range(len(ids_out)):
        #     if 'MusicNet' in ids_out[i]:
        #         ids_out[i] = 'MusicNet'
        # self.ids = list(set(ids_out))
        self.ids_map = {pth.split('/')[-1]: pth.split('/')[-2].split('#')[0] for pth in ids_in}
        for k, v in self.ids_map.items():
            # print('k, v', k, v)
            if 'MusicNet' in v:
                self.ids_map[k] = 'MusicNet'
            elif '1 Bach - Flute sonata in B minor BWV 1030 - Root and Van Delft' in v:
                self.ids_map[k] = 'Flute Sonata 1030 A'
                print('flute a')
            elif '2 BWV 1030 - Flute Sonata in B Minor' in v:
                self.ids_map[k] = 'Flute Sonata 1030 B'
                print('flute b')
            elif different_id_performances is not None and any([elem in v for elem in different_id_performances]):
                self.ids_map[k] = k.split('#')[0]
                print('id identity mapping', k, self.ids_map[k])
            else:
                print('mapping unchanged', k, v)
        self.ids = list(set(self.ids_map.values()))


    def map_id(self, pth):
        # print('map id', pth)
        pth_id = self.ids_map[pth.split('/')[-1]]
        return self.ids.index(pth_id)



class EMDATASET_ALIGNED(EMDATASET):
    def load_pts(self, files):
        self.pts = {}
        print('loading pts...')
        for flac, tsv in tqdm(files):
            print('flac, tsv', flac, tsv)
            if os.path.isfile(self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt')):
                self.pts[flac] = torch.load(self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt'))
            else:
                if '#0' not in flac:
                    orig_flac = flac.split('/')[-1].split('#')[0] + '#0.flac'
                    midi_path = self.labels_path + '/' + orig_flac.replace('.flac', '_BEST_label_.mid')
                    if not os.path.exists(midi_path):
                        midi_path = midi_path.replace('_BEST_label_.mid', '_BEST_alignment_.mid')
                        if not os.path.exists(midi_path):
                            print('skipping midi file shifted', midi_path)
                            continue
                        else:
                            print('found for augmented 2')
                    else:
                        print('found for augmented')


                audio, sr = soundfile.read(flac, dtype='int16')
                audio = audio.astype(float)
                if len(audio.shape) == 2:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.int16)
                assert sr == SAMPLE_RATE
                audio = torch.ShortTensor(audio)
                if '#0' not in flac:
                    assert '#' in flac
                    data = {'audio': audio}
                    self.pts[flac] = data
                    torch.save(data, self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt').replace('.mp3', '.pt'))
                    continue
                aligned_tsv_pth = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.tsv')
                if os.path.isfile(aligned_tsv_pth):
                    print('found tsv')
                    midi = np.loadtxt(aligned_tsv_pth, delimiter='\t', skiprows=1)
                else:
                    print('making tsv')
                    midi_path = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '_BEST_label_.mid')
                    print('making tsv from', midi_path)
                    if not os.path.exists(midi_path):
                        midi_path = midi_path.replace('_BEST_label_.mid', '_BEST_alignment_.mid')
                        print('replaced best label for best alignment')
                        if not os.path.exists(midi_path):
                            print('skipping midi file', midi_path)
                            continue

                    midi = parse_midi_multi(midi_path)
                    # np.savetxt(aligned_tsv_pth, midi,
                    #            fmt='%1.6f', delimiter='\t', header='onset,offset,note,velocity')
                aligned_label, aligned_velocity = midi_to_frames_with_vel(midi, self.instruments, conversion_map=self.conversion_map)
                data = dict(path=self.labels_path + '/' + flac.split('/')[-1],
                            audio=audio, label=aligned_label, velocity=aligned_velocity)
                torch.save(data, self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt').replace('.mp3', '.pt'))
                self.pts[flac] = data

    def update_pts(self):
        print('there are', len(self.pts), 'pts')
        for flac, data in tqdm(self.pts.items()):
            if 'label' not in data:
                continue
            aligned_onsets = (data['label'] >= 3).numpy()
            aligned_frames = (data['label'] >= 2).float().numpy()
            frame_pos_label = np.zeros(aligned_frames.shape, dtype=float)

            for t, f in zip(*aligned_onsets.nonzero()):
                t_end = t
                while t_end < len(aligned_frames) and aligned_frames[t_end, f]:
                    t_end += 1
                num_pts = t_end - t if t_end - t_end >= 0 else 1
                curr_pos_encoding = np.linspace(1., 0., num_pts)
                if t_end - t < 0:
                    print('pos short note')
                frame_pos_label[t: t_end, f] = curr_pos_encoding
            data['frame_pos_label'] = torch.from_numpy(frame_pos_label).float()

    def get_ids(self, different_id_performances=None):
        ids_in = list(self.pts.keys())
        print('ids in', ids_in)
        # ids_out = [pth.split('/')[-2].split('#')[0] for pth in ids_in]
        # for i in range(len(ids_out)):
        #     if 'MusicNet' in ids_out[i]:
        #         ids_out[i] = 'MusicNet'
        # self.ids = list(set(ids_out))
        self.ids_map = {pth.split('/')[-1]: pth.split('/')[-2].split('#')[0] for pth in ids_in}
        for k, v in self.ids_map.items():
            # print('k, v', k, v)
            if 'MusicNet' in v:
                self.ids_map[k] = 'MusicNet'
            elif '1 Bach - Flute sonata in B minor BWV 1030 - Root and Van Delft' in v:
                self.ids_map[k] = 'Flute Sonata 1030 A'
                print('flute a')
            elif '2 BWV 1030 - Flute Sonata in B Minor' in v:
                self.ids_map[k] = 'Flute Sonata 1030 B'
                print('flute b')
            elif different_id_performances is not None and any([elem in v for elem in different_id_performances]):
                self.ids_map[k] = k.split('#')[0]
                print('id identity mapping', k, self.ids_map[k])
            else:
                print('mapping unchanged', k, v)
        self.ids = list(set(self.ids_map.values()))
        for a, b in self.ids_map.items():
            print('AB', a, b)


    def map_id(self, pth):
        # print('map id', pth)
        pth_id = self.ids_map[pth.split('/')[-1]]
        return self.ids.index(pth_id)


class EMDATASET_SLICED(EMDATASET_ALIGNED):
    def __init__(self,
                 audio_path=None,
                 labels_path='LabelsSliced',
                 groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE,
                 instrument_map=None, update_instruments=False, transcriber=None,
                 conversion_map=None, shift_range=(-5, 6), hop_length=HOP_LENGTH,
                 max_parts=False):
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.groups = groups
        self.conversion_map = conversion_map
        self.shift_range = shift_range
        self.hop_length = hop_length
        self.max_parts = max_parts
        self.file_list = self.files(self.groups)
        print('sliced file list', len(self.file_list))
        if instrument_map is None:
            self.get_instruments()
        else:
            self.instruments = instrument_map
            if update_instruments:
                self.add_instruments()
        self.transcriber = transcriber
        self.load_pts(self.file_list)
        print('sliced pts', len(self.pts))
        self.data = [(el, None) for el in self.pts]
        print('sliced data', len(self.data))
        random.shuffle(self.data)

    def __getitem__(self, index):
        data = self.load(*self.data[index])
        result = dict(path=data['path'])
        midi_length = len(data['label'])
        n_steps = self.sequence_length // self.hop_length

        step_begin = self.random.randint(midi_length - n_steps)
        step_end = step_begin + n_steps
        begin = step_begin * self.hop_length
        end = begin + self.sequence_length
        result['audio'] = data['audio'][begin:end]
        diff = self.sequence_length - len(result['audio'])
        result['audio'] = torch.cat((result['audio'], torch.zeros(diff, dtype=result['audio'].dtype)))
        result['audio'] = result['audio'].to(self.device)
        result['label'] = data['label'][step_begin:step_end, ...]
        result['label'] = result['label'].to(self.device)
        if 'velocity' in data:
            result['velocity'] = data['velocity'][step_begin:step_end, ...].to(self.device)
            result['velocity'] = result['velocity'].float() / 128.

        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()

        if 'onset_mask' in data:
            result['onset_mask'] = data['onset_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_mask' in data:
            result['frame_mask'] = data['frame_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_pos_label' in data:
            result['frame_pos_label'] = data['frame_pos_label'][step_begin:step_end, ...].to(self.device).float()

        # print('get item label', result['label'].shape)
        shape = result['frame'].shape
        keys = N_KEYS
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        # frame and offset currently do not differentiate between instruments,
        # so we compress them across instrument and save a copy of the original,
        # as 'big_frame' and 'big_offset'
        result['big_frame'] = result['frame']
        result['frame'], _ = result['frame'].reshape(new_shape).max(axis=-2)
        result['big_offset'] = result['offset']
        result['offset'], _ = result['offset'].reshape(new_shape).max(axis=-2)

        return result

    def files(self, groups, skip_keyword=['I. Overture', 'BWV 1067 - 1. Ouverture', 'BWV 1068 - 1. Ouverture']):
        self.path = '/disk4/ben/PerformanceNet-master/Museopen16_flac'
        if not os.path.exists(self.path):
            self.path = 'Museopen16_flac'
        # self.path = '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen16AUG'

        tsvs_path = '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen_TSV_multi'
        if not os.path.exists(tsvs_path):
            tsvs_path = 'Museopen_TSV_multi'
        res = []
        ### Wind ensembles:
        good_ids = list(range(2075, 2084))

        for group in groups:
            try:
                tsvs = os.listdir(tsvs_path + '/' + group)
            except:
                tsvs = os.listdir(tsvs_path.split('/')[-1] + '/' + group)
            tsvs = sorted(tsvs)
            for shft in range(self.shift_range[0], self.shift_range[1]):
                curr_fls_pth = self.path + '/' + group + '#{}'.format(shft)
                fls = os.listdir(curr_fls_pth)
                fls = [fl for fl in fls if fl.split('/')[-1].split('#')[0] not in ['2194', '2211', '2227', '2230', '2292', '2305', '2310']]
                fls = sorted(fls)
                cnt = 0
                for f, t in zip(fls, tsvs):
                    print('ft', f, t)
                    # if cnt > 3:
                    #     print('breaking at', cnt)
                    #     break
                    cnt += 1
                    # #### MusicNet
                    if 'MusicNet' in group:
                        if all([str(elem) not in f for elem in good_ids]):
                            continue
                    if skip_keyword is not None and any([elem in f for elem in skip_keyword]):
                        print('skipping', f)
                        continue
                    if not os.path.exists(self.labels_path + '/' + f.split('/')[-1].replace('.flac', '$$$PART0.pt')):
                        print('not in sliced label path')
                        continue
                    print('adding', f, t)
                    res.append((curr_fls_pth + '/' + f, tsvs_path + '/' + group + '/' + t))
        return res

    def load_pts(self, files):
        self.pts = {}
        print('loading sliced pts...')
        for flac, tsv in tqdm(files):
            print('flac, tsv', flac, tsv)
            parts = glob(self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '*.pt'), recursive=True)
            print('parts:', len(parts))
            for p, part in enumerate(parts):
                if self.max_parts and p > self.max_parts:
                    continue
                n_part = int(part.split('$$$PART')[1].replace('.pt', ''))
                self.pts[flac.replace('.flac', '$$$PART{}.flac'.format(n_part))] = torch.load(part)

    def update_pts_reduce(self, k=2, pitch_only=False):
        print('there are', len(self.pts), 'pts')
        for flac, data in tqdm(self.pts.items()):
            if 'label' not in data:
                continue
            data['label'] = pool_k(data['label'], k=2)
            if pitch_only:
                # print('pitch only label shape', data['label'].shape)
                data['label'] = data['label'][:, -N_KEYS:]
                # print('pitch only label shape after', data['label'].shape)

    def load(self, audio_path, tsv_path):
        data = self.pts[audio_path]
        if len(data['audio'].shape) > 1:
            data['audio'] = (data['audio'].float().mean(dim=-1)).short()
        if 'label' in data:
            return data
        else:
            f = audio_path.split('/')[-1]
            shift = int(f.split('#')[1].split('$$$')[0])
            assert shift != 0
            orig = audio_path.replace('#{}'.format(shift), '#0')
            res = {}
            res['label'] = shift_label(self.pts[orig]['label'], int(shift))
            res['path'] = audio_path
            res['audio'] = data['audio']
            if 'synth' in data:
                res['synth'] = data['synth']
            if 'velocity' in self.pts[orig]:
                res['velocity'] = shift_label(self.pts[orig]['velocity'], int(shift))
            if 'onset_mask' in self.pts[orig]:
                res['onset_mask'] = shift_label(self.pts[orig]['onset_mask'], int(shift))
            if 'frame_mask' in self.pts[orig]:
                res['frame_mask'] = shift_label(self.pts[orig]['frame_mask'], int(shift))
            if 'frame_pos_label' in self.pts[orig]:
                res['frame_pos_label'] = shift_label(self.pts[orig]['frame_pos_label'], int(shift))
            return res

class MUSICNET_EMDATASET_SLICED(EMDATASET_SLICED):
    def files(self, groups, skip_keyword=['I. Overture', 'BWV 1067 - 1. Ouverture', 'BWV 1068 - 1. Ouverture']):
        self.path = '/disk4/ben/PerformanceNet-master/Museopen16_flac'
        if not os.path.exists(self.path):
            self.path = 'Museopen16_flac'
        # self.path = '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen16AUG'
        tsvs_path = '/disk3/ben/onsets_and_frames/onsets-and-frames-master/Museopen_TSV_multi'
        if not os.path.exists(tsvs_path):
            tsvs_path = 'Museopen_TSV_multi'
        res = []
        train_f = open('/home/dcor/benmaman/MusicNet/my_train.txt', 'r')
        musicnet_my_train_set = train_f.read().split()
        train_f.close()

        ### Wind ensembles:
        for group in groups:
            # try:
            #     tsvs = os.listdir(tsvs_path + '/' + group)
            # except:
            #     tsvs = os.listdir(tsvs_path.split('/')[-1] + '/' + group)
            # tsvs = sorted(tsvs)
            for shft in range(self.shift_range[0], self.shift_range[1]):
                curr_fls_pth = self.path + '/' + group + '#{}'.format(shft)
                fls = os.listdir(curr_fls_pth)
                fls = [fl for fl in fls if fl.split('/')[-1].split('#')[0] not in ['2194', '2211', '2227', '2230', '2292', '2305', '2310']]

                print('before train filter', len(fls))
                fls = [fl for fl in fls if fl.split('/')[-1].split('#')[0] in musicnet_my_train_set]


                # debug_subset = ['1790', '1817', '1932', '2079', '2081', '2166', '2219', '2282', '2330', '2383']
                # fls = [fl for fl in fls if any(ds in fl for ds in debug_subset)]
                # print('shift', shft, 'fls', fls)

                print('after train filter', len(fls))

                # fls = [fl for fl in fls if int(fl.split('/')[-1].split('#')[0]) < 1800]
                # print('only under 1800')

                fls = sorted(fls)
                cnt = 0
                for f in fls:
                    print('ft', f)
                    # if cnt > 3:
                    #     print('breaking at', cnt)
                    #     break
                    cnt += 1
                    # #### MusicNet
                    if skip_keyword is not None and any([elem in f for elem in skip_keyword]):
                        print('skipping', f)
                        continue
                    if not os.path.exists(self.labels_path + '/' + f.split('/')[-1].replace('.flac', '$$$PART0.pt')):
                        print('not in sliced label path')
                        continue
                    print('adding', f)
                    # res.append((curr_fls_pth + '/' + f, tsvs_path + '/' + group + '/' + t))
                    res.append((curr_fls_pth + '/' + f, None))
        print('all files', len(res))
        return res

    def get_ids_musicnet(self, different_id_performances=None):
        ids_in = list(self.pts.keys())
        # print('pts keys', list(self.pts.keys()))
        # for kd, d in self.pts.items():
        #     print('kd', kd, 'd keys', list(d.keys()))
        # ids_in = [el['path'] for el in self.pts.values()]

        print('ids in', ids_in)
        # ids_out = [pth.split('/')[-2].split('#')[0] for pth in ids_in]
        # for i in range(len(ids_out)):
        #     if 'MusicNet' in ids_out[i]:
        #         ids_out[i] = 'MusicNet'
        # self.ids = list(set(ids_out))
        self.ids_map = {pth.split('/')[-1]: pth.split('/')[-2] for pth in ids_in}
        print('get ids musicnet')
        for k, v in self.ids_map.items():
            print('items k, v', k, v)
            shift = v.split('#')[1]
            assert int(shift) in list(range(-5, 6))
            self.ids_map[k] = k.split('#')[0]
            # self.ids_map[k] = k.split('#')[0] + '#{}'.format(shift)
            print('id identity mapping', k, self.ids_map[k])
        self.ids = list(set(self.ids_map.values()))
        print('self ids', self.ids)

class EMDATASET_ALIGNED2(EMDATASET):
    def __init__(self,
                 audio_path='NoteEM_audio',
                 labels_path='NoteEm_labels',
                 groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE,
                 instrument_map=None, update_instruments=False, transcriber=None,
                 conversion_map=None, shift_range=(-5, 6)):
        self.audio_path = audio_path
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.groups = groups
        self.conversion_map = conversion_map
        self.shift_range = shift_range
        self.file_list = self.files(self.groups)
        if instrument_map is None:
            self.get_instruments()
        else:
            self.instruments = instrument_map
            if update_instruments:
                self.add_instruments()
        self.transcriber = transcriber
        self.load_pts(self.file_list)
        self.data = []
        print('Reading files...')
        for input_files in tqdm(self.file_list):
            data = torch.load(self.pts[input_files[0]])
            audio_len = len(data['audio'])
            minutes = audio_len // (SAMPLE_RATE * 60)
            copies = minutes
            # copies = 1
            for _ in range(copies):
                self.data.append(input_files)
        random.shuffle(self.data)

    def load(self, audio_path, tsv_path):
        data_path = self.pts[audio_path]
        data = torch.load(data_path)
        if len(data['audio'].shape) > 1:
            data['audio'] = (data['audio'].float().mean(dim=-1)).short()
        if 'label' in data:
            print('data keys', list(data.keys()))
            return data
        else:
            piece, part = audio_path.split('/')[-2:]
            piece_split = piece.split('#')
            if len(piece_split) == 2:
                piece, shift1 = piece_split
            else:
                piece, shift1 = '#'.join(piece_split[:2]), piece_split[-1]
            part_split = part.split('#')
            if len(part_split) == 2:
                part, shift2 = part_split
            else:
                part, shift2 = '#'.join(part_split[:2]), part_split[-1]
            shift2, _ = shift2.split('.')
            assert shift1 == shift2
            shift = shift1
            assert shift != 0
            orig = audio_path.replace('#{}'.format(shift), '#0')
            data_orig = torch.load(self.pts[orig])
            res = {}
            res['label'] = shift_label(data_orig['label'], int(shift))
            res['path'] = audio_path
            res['audio'] = data['audio']
            if 'synth' in data:
                res['synth'] = data['synth']
            if 'velocity' in data_orig:
                res['velocity'] = shift_label(data_orig['velocity'], int(shift))
            if 'onset_mask' in data_orig:
                res['onset_mask'] = shift_label(data_orig['onset_mask'], int(shift))
            if 'frame_mask' in data_orig:
                res['frame_mask'] = shift_label(data_orig['frame_mask'], int(shift))
            if 'frame_pos_label' in data_orig:
                res['frame_pos_label'] = shift_label(data_orig['frame_pos_label'], int(shift))
            return res




    def load_pts(self, files):
        self.pts = {}
        print('loading pts...')
        for flac, tsv in tqdm(files):
            print('flac, tsv', flac, tsv)
            if os.path.isfile(self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt')):
                # self.pts[flac] = torch.load(self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt'))
                self.pts[flac] = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt')

            else:
                if '#0' not in flac:
                    orig_flac = flac.split('/')[-1].split('#')[0] + '#0.flac'
                    midi_path = self.labels_path + '/' + orig_flac.replace('.flac', '_BEST_label_.mid')
                    if not os.path.exists(midi_path):
                        midi_path = midi_path.replace('_BEST_label_.mid', '_BEST_alignment_.mid')
                        if not os.path.exists(midi_path):
                            print('skipping midi file shifted', midi_path)
                            continue
                        else:
                            print('found for augmented 2')
                    else:
                        print('found for augmented')


                audio, sr = soundfile.read(flac, dtype='int16')
                audio = audio.astype(float)
                if len(audio.shape) == 2:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.int16)
                assert sr == SAMPLE_RATE
                audio = torch.ShortTensor(audio)
                if '#0' not in flac:
                    assert '#' in flac
                    data = {'audio': audio}
                    # self.pts[flac] = data
                    self.pts[flac] = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt').replace('.mp3', '.pt')
                    torch.save(data, self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt').replace('.mp3', '.pt'))
                    continue
                aligned_tsv_pth = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.tsv')
                if os.path.isfile(aligned_tsv_pth):
                    print('found tsv')
                    midi = np.loadtxt(aligned_tsv_pth, delimiter='\t', skiprows=1)
                else:
                    print('making tsv')
                    midi_path = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '_BEST_label_.mid')
                    print('making tsv from', midi_path)
                    if not os.path.exists(midi_path):
                        midi_path = midi_path.replace('_BEST_label_.mid', '_BEST_alignment_.mid')
                        print('replaced best label for best alignment')
                        if not os.path.exists(midi_path):
                            print('skipping midi file', midi_path)
                            continue

                    midi = parse_midi_multi(midi_path)
                    # np.savetxt(aligned_tsv_pth, midi,
                    #            fmt='%1.6f', delimiter='\t', header='onset,offset,note,velocity')
                aligned_label, aligned_velocity = midi_to_frames_with_vel(midi, self.instruments, conversion_map=self.conversion_map)
                data = dict(path=self.labels_path + '/' + flac.split('/')[-1],
                            audio=audio, label=aligned_label, velocity=aligned_velocity)
                torch.save(data, self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt').replace('.mp3', '.pt'))
                # self.pts[flac] = data
                self.pts[flac] = self.labels_path + '/' + flac.split('/')[-1].replace('.flac', '.pt').replace('.mp3', '.pt')



    def update_pts(self):
        print('there are', len(self.pts), 'pts')
        for flac, data in tqdm(self.pts.items()):
            if 'label' not in data:
                continue
            aligned_onsets = (data['label'] >= 3).numpy()
            aligned_frames = (data['label'] >= 2).float().numpy()
            frame_pos_label = np.zeros(aligned_frames.shape, dtype=float)

            for t, f in zip(*aligned_onsets.nonzero()):
                t_end = t
                while t_end < len(aligned_frames) and aligned_frames[t_end, f]:
                    t_end += 1
                num_pts = t_end - t if t_end - t_end >= 0 else 1
                curr_pos_encoding = np.linspace(1., 0., num_pts)
                if t_end - t < 0:
                    print('pos short note')
                frame_pos_label[t: t_end, f] = curr_pos_encoding
            data['frame_pos_label'] = torch.from_numpy(frame_pos_label).float()




    def get_ids(self, different_id_performances=None):
        ids_in = list(self.pts.keys())
        print('ids in', ids_in)
        # ids_out = [pth.split('/')[-2].split('#')[0] for pth in ids_in]
        # for i in range(len(ids_out)):
        #     if 'MusicNet' in ids_out[i]:
        #         ids_out[i] = 'MusicNet'
        # self.ids = list(set(ids_out))
        self.ids_map = {pth.split('/')[-1]: pth.split('/')[-2].split('#')[0] for pth in ids_in}
        for k, v in self.ids_map.items():
            # print('k, v', k, v)
            if 'MusicNet' in v:
                self.ids_map[k] = 'MusicNet'
            elif '1 Bach - Flute sonata in B minor BWV 1030 - Root and Van Delft' in v:
                self.ids_map[k] = 'Flute Sonata 1030 A'
                print('flute a')
            elif '2 BWV 1030 - Flute Sonata in B Minor' in v:
                self.ids_map[k] = 'Flute Sonata 1030 B'
                print('flute b')
            elif different_id_performances is not None and any([elem in v for elem in different_id_performances]):
                self.ids_map[k] = k.split('#')[0]
                print('id identity mapping', k, self.ids_map[k])
            else:
                print('mapping unchanged', k, v)
        self.ids = list(set(self.ids_map.values()))


    def map_id(self, pth):
        # print('map id', pth)
        pth_id = self.ids_map[pth.split('/')[-1]]
        return self.ids.index(pth_id)


class EMDATASET_SYNTH(EMDATASET):
    def __getitem__(self, index):
        data = self.load(*self.data[index])
        result = dict(path=data['path'])
        midi_length = len(data['label'])
        n_steps = self.sequence_length // HOP_LENGTH
        step_begin = self.random.randint(midi_length - n_steps)
        step_end = step_begin + n_steps
        begin = step_begin * HOP_LENGTH
        end = begin + self.sequence_length
        result['audio'] = data['audio'][begin:end]
        diff = self.sequence_length - len(result['audio'])
        result['audio'] = torch.cat((result['audio'], torch.zeros(diff, dtype=result['audio'].dtype)))
        result['audio'] = result['audio'].to(self.device)

        # synth_key = random.choice(list(data['synth'].keys()))
        # result['synth'] = data['synth'][synth_key][:, begin: end]


        synth_mask = np.random.uniform(0., 1., len(self.instruments)) > 0.5
        synth_key1 = np.argwhere(synth_mask)[:, 0]
        synth_key2 = np.argwhere(~synth_mask)[:, 0]
        synth = data['synth'][:, begin: end]
        synth1 = synth[synth_key1, :].sum(dim=0).unsqueeze(0)
        synth2 = synth[synth_key2, :].sum(dim=0).unsqueeze(0)
        result['synth'] = torch.cat([synth1, synth2], dim=0)
        # diff = self.sequence_length - result['synth'].shape[-1]
        # result['synth'] = torch.cat((result['synth'], torch.zeros(diff, dtype=result['synth'].dtype)))
        result['synth'] = result['synth'].to(self.device)


        result['label'] = data['label'][step_begin:step_end, ...]
        result['label'] = result['label'].to(self.device)
        if 'velocity' in data:
            result['velocity'] = data['velocity'][step_begin:step_end, ...].to(self.device)
            result['velocity'] = result['velocity'].float() / 128.

        result['audio'] = result['audio'].float()
        result['audio'] = result['audio'].div_(32768.0)

        result['synth'] = result['synth'].float()
        result['synth'] = result['synth'].div_(32768.0)

        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()

        if 'onset_mask' in data:
            result['onset_mask'] = data['onset_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_mask' in data:
            result['frame_mask'] = data['frame_mask'][step_begin:step_end, ...].to(self.device).float()
        if 'frame_pos_label' in data:
            result['frame_pos_label'] = data['frame_pos_label'][step_begin:step_end, ...].to(self.device).float()


        shape = result['frame'].shape
        keys = N_KEYS
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        # frame and offset currently do not differentiate between instruments,
        # so we compress them across instrument and save a copy of the original,
        # as 'big_frame' and 'big_offset'
        result['big_frame'] = result['frame']
        result['frame'], _ = result['frame'].reshape(new_shape).max(axis=-2)
        result['big_offset'] = result['offset']
        result['offset'], _ = result['offset'].reshape(new_shape).max(axis=-2)
        return result

    def create_synth(self):
        soundfont = '/disk3/ben/Soundfonts/MuseScore_General.sf2'
        # soundfont = '/disk3/ben/Soundfonts/FluidR3_GM.sf2'

        fl = fluidsynth.Synth(samplerate=SAMPLE_RATE)
        sfid = fl.sfload(soundfont)
        for audio_path, data in self.pts.items():
            if 'unaligned_label' not in data:
                continue
            label, velocity = data['label'], data['velocity']
            print('synth', audio_path)
            for shift in range(self.shift_range[0], self.shift_range[1]):
                print('shift', shift)
                shifted_pth = audio_path.replace('#0', '#{}'.format(shift))
                shifted_pt_pth = 'SYNTHED/' + shifted_pth.split('/')[-1].replace('.flac', '_synthed.pt')
                if os.path.isfile(shifted_pt_pth):
                    self.pts[shifted_pth]['synth'] = torch.load(shifted_pt_pth)
                    continue
                shifted_label = shift_label(label, shift=shift)[:, : -N_KEYS]
                shifted_vel = shift_label(velocity, shift=shift)[:, : -N_KEYS]
                synthed = []
                for curr_inst in self.instruments:
                    curr_comp = set(self.instruments) - {curr_inst}
                    curr_label = shifted_label.clone()
                    for inst in curr_comp:
                        inst_index = self.instruments.index(inst)
                        curr_label[:, inst_index * N_KEYS: (inst_index + 1) * N_KEYS] = 0
                    s = render_frames(curr_label.numpy(), shifted_vel.numpy(),
                                      instruments=self.instruments,
                                      shift=0, SPF=HOP_LENGTH,
                                      fl=fl, sfid=sfid)
                    s = np.vstack((s[:: 2], s[1:: 2]))
                    s = librosa.to_mono(s.astype(float)).astype(np.int16)
                    # os.makedirs('test_synth', exist_ok=True)
                    # soundfile.write('test_synth/' + audio_path.split('/')[-1] + '{}_{}.flac'.format(shift, curr_inst), s, SAMPLE_RATE)
                    audio = torch.ShortTensor(s)
                    synthed.append(audio.unsqueeze(0))
                synthed = torch.cat(synthed, dim=0)
                self.pts[shifted_pth]['synth'] = synthed
                torch.save(synthed, shifted_pt_pth)
                # if 'synth' not in self.pts[shifted_pth]:
                #     self.pts[shifted_pth]['synth'] = {}
                # self.pts[shifted_pth]['synth'][curr_inst] = synthed








                # for inst_comb in powerset(self.instruments):
                #     inst_comp = list(set(self.instruments) - set(inst_comb))
                #     synthed = []
                #     for curr_insts, curr_comp in zip([inst_comb, inst_comp], [inst_comp, inst_comb]):
                #         curr_label = shifted_label.clone()
                #         print('curr', curr_insts, curr_comp)
                #         for inst in curr_comp:
                #             inst_index = self.instruments.index(inst)
                #             curr_label[:, inst_index * N_KEYS: (inst_index + 1) * N_KEYS] = 0
                #         s = render_frames(curr_label, shifted_vel,
                #                           instruments=self.instruments,
                #                           shift=0, SPF=HOP_LENGTH,
                #                           fl=fl, sfid=sfid)
                #         s = np.vstack((s[:: 2], s[1:: 2]))
                #         print('s shape', s.shape, s.max(), s.min())
                #         s = librosa.to_mono(s.astype(float)).astype(np.int16)
                #         os.makedirs('test_synth', exist_ok=True)
                #         soundfile.write('test_synth/' + audio_path.split('/')[-1] + '{}_{}.flac'.format(shift, curr_insts), s, SAMPLE_RATE)
                #         audio = torch.ShortTensor(s)
                #         synthed.append(audio.unsqueeze(0))
                #     synthed = torch.cat(synthed, dim=0)
                #     shifted_pth = audio_path.replace('#0', '#{}'.format(shift))
                #     if 'synth' not in self.pts[shifted_pth]:
                #         self.pts[shifted_pth]['synth'] = {}
                #     self.pts[shifted_pth]['synth'][tuple(sorted(inst_comb)), tuple(sorted(inst_comp))] = synthed





