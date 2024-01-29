import argparse
import numpy as np
import torch
torch.set_grad_enabled(False)
import soundfile as sf
import os
from onsets_and_frames.mel import STFT
from onsets_and_frames.denoising_diffusion_t5 import *
import librosa

from eval_set_midis import *
from eval_set_insts import *
from eval_set import *

def main(cfg, cfg_id, model_pth, epoch, cfg_type=1, ckpt_path='runs/', sampling_timesteps=250):
    print('cfg weights', cfg, cfg_id)
    ckpt = ckpt_path + '/' + model_pth + '/model_{}.pt'.format(epoch)
    print('reading from ckpt', ckpt)
    mapping_path = ckpt_path + '/' + model_pth + '/instrument_mapping.pt'
    map_pt = torch.load(mapping_path)
    instrument_map = map_pt['instrument_mapping']
    print('read mapping from pt', instrument_map)
    ids_mapping = torch.load(ckpt_path + '/' + model_pth + '/ids.pt')
    ids_mapping = ids_mapping['ids']
    print('read ids mapping', ids_mapping)

    configs = torch.load(ckpt_path + '/' + model_pth + '/config.pt')
    print('read config:', configs)
    N_FFT = configs['N_FFT']
    N_MELS = configs['N_MELS']
    WINDOW_LENGTH = configs['WINDOW_LENGTH']
    HOP_LENGTH = configs['HOP_LENGTH']
    MEL_FMIN_THIS = configs['MEL_FMIN']
    MEL_FMAX = configs['MEL_FMAX']
    MEL_NORM = configs['MEL_NORM']

    use_strict_log = configs['use_strict_log']
    objective = configs['objective']
    clip_max = configs['clip_max']
    clip_min = configs['clip_min']
    norm_factor = configs['norm_factor']
    image_size = configs['image_size']

    print('read config parameters')

    curr_pth = 'sampling/' + '{}_epoch_{}_cfgs{}{}_{}_steps{}'.format(model_pth, epoch, ('TYPETWO' if cfg_type == 2 else ''), cfg, cfg_id, sampling_timesteps) + '/sampling/'
    curr_pth_long = curr_pth + 'long'
    midis_path_to_save = curr_pth_long + '/midis'
    unquantized_midis_path_to_save = curr_pth_long + '/unquantized_midis'

    os.makedirs(curr_pth_long, exist_ok=True)
    os.makedirs(midis_path_to_save, exist_ok=True)
    os.makedirs(unquantized_midis_path_to_save, exist_ok=True)

    print('checking all midis and ids exist...')
    for midi_pth, performance_id in midis_and_performances:
        if not os.path.isfile(midi_pth):
            print('midi pth does not exist', midi_pth)
            assert os.path.isfile(midi_pth)
        if performance_id not in ids_mapping:
            print(performance_id, 'not in')
            assert performance_id in ids_mapping
    print('done.')


    for midi_pth, performance_id in midis_and_performances:
        curr_force_conversion = get_conversion(performance_id)
        print('curr conversion', curr_force_conversion)
        big_spec_path = curr_pth_long + '/' + midi_pth.split('/')[-1] + '#long_mel_spec#{}.npy'.format(performance_id)
        if os.path.exists(big_spec_path):
            print('found', big_spec_path)
            print('skipping')
            continue

        midi = parse_midi_multi(midi_pth, force_instrument=None)
        print('input instruments:', set(midi[:, -1].astype(int)))
        alpha_tempo = 1.
        # alpha_tempo = 300 / 256
        midi[:, :2] /= alpha_tempo
        max_time = int(midi[:, 1].max() + 1)
        audio_length = max_time * SAMPLE_RATE
        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        n_channels = len(instrument_map) + 1

        label = torch.zeros(n_steps, n_keys * n_channels, dtype=torch.uint8)
        frame_label = torch.zeros(n_steps, n_keys * n_channels, dtype=torch.float)

        start_time = 1.5

        # Create piano roll from midi (slightly different than midi_to_frames function in onsets_and_frames/midi_utils.py):
        for onset, offset, note, vel, instrument in midi:
            if int(instrument) not in instrument_map and (curr_force_conversion is None or int(instrument) not in curr_force_conversion or curr_force_conversion[int(instrument)] not in instrument_map):
                continue
            if curr_force_conversion is not None:
                if instrument in curr_force_conversion:
                    instrument = curr_force_conversion[instrument]
            f = int(note) - MIN_MIDI
            if f >= n_keys or f < 0:
                continue
            mapped_instrument = int(instrument)
            if int(mapped_instrument) not in instrument_map:
                if conversion_map is not None and int(mapped_instrument) in conversion_map:
                    mapped_instrument = conversion_map[int(mapped_instrument)]
                else:
                    continue
            onset += start_time
            offset += start_time
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)
            chan = instrument_map.index(int(mapped_instrument))
            label[left:onset_right, n_keys * chan + f] = 3
            label[onset_right:frame_right, n_keys * chan + f] = 2
            label[frame_right:offset_right, n_keys * chan + f] = 1

            pitch_chan = len(instrument_map)
            label[left:onset_right, n_keys * pitch_chan + f] = 3
            label[onset_right:frame_right, n_keys * pitch_chan + f] = 2
            label[frame_right:offset_right, n_keys * pitch_chan + f] = 1

            num_pts = frame_right - left if frame_right - left >= 0 else 1
            curr_pos_encoding = np.linspace(1., 0., num_pts)
            if frame_right - onset_right < 0:
                print('short note')
            frame_label[left:frame_right, n_keys * chan + f] = torch.from_numpy(curr_pos_encoding)
            frame_label[left:frame_right, n_keys * pitch_chan + f] = torch.from_numpy(curr_pos_encoding)

        # save copy of midi (temporally quantized - from piano roll):
        f_name_to_save = midi_pth.split('/')[-1] + '#' + performance_id
        onsets_to_save = (label[:, : -n_keys] == 3).cpu().numpy()
        frames_to_save = (label == 2).reshape((label.shape[0], len(instrument_map) + 1, n_keys)).max(dim=1)[0].cpu().numpy()
        p_ret, i_ret, v_ret, inst_ret = frames2midi(midis_path_to_save + '/' + f_name_to_save + '.mid',
                    onsets_to_save,
                    frames_to_save,
                    64 * onsets_to_save, inst_mapping=instrument_map,
                                                    scaling=HOP_LENGTH / SAMPLE_RATE)
        # save copy of original midi (unquantized):
        copy_command = 'cp \"{}\" \"{}\"'.format(midi_pth, unquantized_midis_path_to_save + '/' + f_name_to_save + '.mid')
        os.system(copy_command)

        big_label = label.unsqueeze(0).float().cuda()
        big_frame_label = frame_label.unsqueeze(0).float().cuda()

        model = torch.load(ckpt)#.cuda()
        diffusion_model = GaussianDiffusion(model, image_size=image_size, objective=objective, norm_factor=norm_factor, use_log=False,
                                                    use_strict_log=use_strict_log, p2_loss_weight_gamma=1.,
                                            clip_max=clip_max, clip_min=clip_min, sampling_timesteps=sampling_timesteps
                                            ).cuda()
        model.eval()

        big_inverse = []
        big_mel_np = []

        os.makedirs(curr_pth, exist_ok=True)

        batch_size = 16
        batch = []

        curr_idx = 0
        reached_end = False
        diff_steps = 1000
        use_frames = True

        slice_len = diffusion_model.image_size[0]
        scaling, margin = 1, 0
        overlap = 32 # number of frames in overlap, over which repeated interpolation is applied
        half_overlap = overlap // 2
        n_slices = (n_steps - overlap) // (slice_len - overlap)
        if n_slices * (slice_len - overlap) + overlap < n_steps:
            n_slices += 1

        for slice in range(n_slices):
            begin = slice * (slice_len - overlap)
            end = begin + slice_len
            end = min(end, big_label.shape[1])
            label = big_label[:, begin * scaling - margin: end * scaling + margin, ...]
            frames = big_frame_label[:, begin * scaling - margin: end * scaling + margin, ...]
            if label.shape[1] < diffusion_model.image_size[0]:
                reached_end = True
                end_pad = diffusion_model.image_size[0] - label.shape[1]
                label = torch.nn.functional.pad(label, (0, 0, 0, end_pad))
                frames = torch.nn.functional.pad(frames, (0, 0, 0, diffusion_model.image_size[0] - frames.shape[1]))
            elif end * scaling + margin == big_label.shape[1]:
                reached_end = True
                end_pad = diffusion_model.image_size[0] - label.shape[1]
                label = torch.nn.functional.pad(label, (0, 0, 0, end_pad))
                frames = torch.nn.functional.pad(frames, (0, 0, 0, diffusion_model.image_size[0] - frames.shape[1]))


            onsets = label == 3
            onsets = onsets.float()
            frames = use_frames * frames
            notes = torch.cat((onsets.permute((0, 2, 1)), frames.permute((0, 2, 1))), dim=1)
            batch.append(notes)
            if len(batch) < batch_size and not reached_end:
                continue

            notes = torch.cat(batch, dim=0)
            ids = torch.Tensor([ids_mapping.index(performance_id) for _ in range(notes.shape[0])]).long().cuda()
            ids = F.one_hot(ids, num_classes=len(ids_mapping)).float()

            y_preds = diffusion_model.ddim_sample_multi_cfg((notes.shape[0], N_MELS_THIS, image_size[0]), notes,
                                                  ids=ids,
                                                  clip_denoised=True, cfg=cfg, cfg_id=cfg_id,
                                                      overlap=overlap,
                                                            cfg_type=cfg_type
                                                  )
            y_preds = y_preds.detach().cpu().numpy()
            batch = []

            for i_spec, spec in enumerate(y_preds):
                curr_idx += 1

                left = 0 if curr_idx == 1 else half_overlap
                right = -end_pad if reached_end and i_spec == y_preds.shape[0] - 1 else -half_overlap
                spec_out = np.exp(spec) if use_strict_log else spec
                spec_out = np.clip(spec_out, a_min=1e-5, a_max=1e8)
                spec_out = spec_out[:, left: right]
                if diffusion_model.objective == 'pred_x0' or diff_steps > 1:
                    if i_spec == 2:
                        print('inverting...')
                        inverse = librosa.feature.inverse.mel_to_audio(spec_out, sr=SAMPLE_RATE, n_fft=N_FFT, win_length=WINDOW_LENGTH,
                                                                       power=1., hop_length=HOP_LENGTH, htk=True, fmin=MEL_FMIN_THIS, fmax=MEL_FMAX, norm=MEL_NORM)
                        print('done inverting.')
                        big_inverse.append(inverse)
                        sf.write(curr_pth + '/' + midi_pth.split('/')[-1] + '#{}#{}.flac'.format(curr_idx, performance_id), inverse, SAMPLE_RATE,
                                 format='flac', subtype='PCM_24')
                mel_np = spec[:, left: right]
                big_mel_np.append(mel_np)
            if reached_end:
                break

        big_inverse = np.concatenate(big_inverse)
        sf.write(curr_pth_long + '/' + midi_pth.split('/')[-1] + '#long#{}.flac'.format(performance_id), big_inverse, SAMPLE_RATE,
                 format='flac', subtype='PCM_24')
        big_spec = np.concatenate(big_mel_np, axis=1)
        np.save(big_spec_path, big_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-cfg')
    # parser.add_argument('-cfgid')
    # default_synthesizer_ckpt = '/home/dcor/benmaman/PerformanceConditioning/runs/mel_large_classical_t5film_448-231111-095446 DONE/model_3.pt'
    parser.add_argument('-model_pth', default='mel_large_classical_t5film_448-231111-095446 DONE')
    parser.add_argument('-epoch', default='3')
    parser.add_argument('-cfg_type', default='1')
    args = parser.parse_args()


    cfgs = [(1.25, 1.25)]

    for cfg, cfg_id in cfgs:
        main(cfg=float(cfg), cfg_id=float(cfg_id), model_pth=args.model_pth, epoch=int(args.epoch), cfg_type=int(args.cfg_type),
             ckpt_path='/home/dcor/benmaman/PerformanceConditioning/runs/', sampling_timesteps=50)