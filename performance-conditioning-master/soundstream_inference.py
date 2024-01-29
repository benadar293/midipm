import math
import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_hub as hub
import soundfile
import numpy as np
import librosa
# from mel_norm import *
import numpy as np
import os

# module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')
# module = hub.KerasLayer('/disk4/ben/PerformanceNet-master/soundstream_ckpt')
module = hub.KerasLayer('/home/dcor/benmaman/PerformanceNet-master/soundstream_ckpt')



SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8



# src_path = '/home/dcor/benmaman/performance-conditioning-master/sampling/mel_large_classical_t5film_448-231111-095446 DONE_epoch_3_cfgs1.25_1.25_steps25/sampling/long/'
src_path = '/home/dcor/benmaman/performance-conditioning-master/sampling/mel_large_classical_t5film_448-231111-095446 DONE_epoch_3_cfgs1.25_1.25_steps50/sampling/long/'
choices = [
    # 'italian'
    # 'beethoven_symphony_5_1',
    # 'messiah_hallelujah'
           ]
# Load a music sample from the GTZAN dataset.
# Convert an example from int to float.
out_path = src_path + '/soundstream'
os.makedirs(out_path, exist_ok=True)
audio_pths = [src_path + '/' + elem for elem in os.listdir(src_path) if elem.endswith('.npy')]
for audio_pth in audio_pths:
    if choices and all([el not in audio_pth for el in choices]):
        continue
    curr_audio_out_path = out_path + '/' + audio_pth.split('/')[-1].replace('.npy', '.flac')
    if os.path.isfile(curr_audio_out_path):
        print('found file, skipping')
        continue
    print('inverting', audio_pth)
    spectrogram = np.load(audio_pth)
    print('clipping')
    spectrogram = np.clip(spectrogram, a_min=np.log(1e-5), a_max=np.log(1e8))
    print('spectrogram shape', spectrogram.shape)
    # spectrogram += np.log(10)
    MAX_LEN = 300 * 50
    audio_outs = []
    parts = spectrogram.shape[1] // MAX_LEN + (spectrogram.shape[1] % MAX_LEN != 0)
    print('parts:', parts)
    for i in range(parts):
        print('part', i)
        part = spectrogram[:, i * MAX_LEN: (i + 1) * MAX_LEN]
        part = tf.convert_to_tensor(part.T.astype(np.float32))
        part = tf.expand_dims(part, axis=0)
        # Reconstruct the audio from a mel-spectrogram using a SoundStream decoder.
        curr_out = module(part).numpy()[0]
        print('curr out shape', curr_out.shape)
        audio_outs.append(curr_out)

    audio_out = np.concatenate(audio_outs)
    print('audio out shape', audio_out.shape)
    soundfile.write(curr_audio_out_path, audio_out, SAMPLE_RATE,
             format='flac', subtype='PCM_24')