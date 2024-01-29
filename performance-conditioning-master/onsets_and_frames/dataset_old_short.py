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