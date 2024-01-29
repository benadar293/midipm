import multiprocessing
import sys

import mido
import numpy as np
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi
from tqdm import tqdm

DRUM_CHANNEL = 9


def parse_midi(path, channels=False, controls=False, use_sustain=True):
    if controls and not channels:
        raise ValueError('controls and not channnels')
    if use_sustain and channels:
        raise ValueError('sustain with channnels not supported yet')
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)
    time = 0
    sustain = False

    events = []
    if controls:
        control_changes = []

    # messages = set([(message.type, message.channel) for message in midi if hasattr(message, 'channel')])
    # for m in messages:
    #     print(m)
    # for m in midi:
    #     # if m.type in ['control_change', 'program_change']:
    #     if m.type == 'program_change':
    #     # if m.type == 'control_change':
    #         print(m)
    if channels:
        instruments = {}
    all_channels = set()
    for message in midi:
        time += message.time
        if hasattr(message, 'channel'):
            if message.channel == DRUM_CHANNEL:
                continue

        if use_sustain:
            if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
                # sustain pedal state has just changed
                sustain = message.value >= 64
                event_type = 'sustain_on' if sustain else 'sustain_off'
                event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
                if channels:
                    event['channel'] = message.channel
                events.append(event)

        if controls and time == 0 and message.type == 'control_change' and message.control != 64:
            control_changes.append((time, message.control, message.value, message.channel))

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note,
                         velocity=velocity, sustain=sustain)
            if channels:
                event['channel'] = message.channel
            events.append(event)

        if channels:
            if hasattr(message, 'channel'):
                all_channels.add(message.channel)
            if message.type == 'program_change':
                instruments[message.channel] = message.program
    if channels:
        for el in all_channels - set(instruments.keys()):
            instruments[el] = 0

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue
        # find the next note_off message
        if not channels:
            try:
                offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])
            except StopIteration:
                print('Stop iteration occured in midi')
                offset = events[-1]
        else:
            offset = next(n for n in events[i + 1:] if (n['note'] == onset['note']
                                                        and n['channel'] == onset['channel'])
                          or n is events[-1])
        # if offset['time'] - onset['time'] > 5:
        #     print(offset['time'] - onset['time'], offset['sustain'])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            if not channels:
                offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])
            else:
                offset = next(n for n in events[offset['index'] + 1:] if (n['type'] == 'sustain_off'
                                                                          and n['channel'] == onset['channel'])
                              or n is events[-1])

        if not channels:
            note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        else:
            note = (onset['time'], offset['time'], onset['note'], onset['velocity'],
                    onset['channel'], instruments[onset['channel']])
        notes.append(note)

    res = np.array(notes)
    if channels:
        if controls:
            control_changes = np.array(control_changes, dtype=np.uint8)
        res = (res, instruments) if not controls else (res, instruments, control_changes)
    return res


def append_track(file, pitches, intervals, velocities):
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        try:
            track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        except Exception as e:
            print('Err Message', 'note_' + event['type'], pitch, velocity, current_tick - last_tick)
            track.append(Message('note_' + event['type'], note=pitch, velocity=max(0, velocity), time=current_tick - last_tick))
            if velocity >= 0:
                raise e
        last_tick = current_tick


def append_track_multi(file, pitches, intervals, velocities, ins, single_ins=False):
    track = MidiTrack()
    file.tracks.append(track)
    chan = len(file.tracks) - 1
    print('chan', chan, 'ins', ins)
    if chan >= DRUM_CHANNEL:
        chan += 1
    if chan > 15:
        print('too many channels, skipping')
        return
    track.append(Message('program_change', channel=chan, program=ins if not single_ins else 0, time=0))
    # if ins != -9:
    #     track.append(Message('program_change', channel=chan, program=ins if not single_ins else 0, time=0))
    # else:
    #     track.append(Message('program_change', channel=DRUM_CHANNEL, program=0, time=0))

    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        try:
            track.append(Message('note_' + event['type'], channel=chan, note=pitch, velocity=velocity, time=current_tick - last_tick))
        except Exception as e:
            print('Err Message', 'note_' + event['type'], pitch, velocity, current_tick - last_tick)
            track.append(Message('note_' + event['type'], channel=chan, note=pitch, velocity=max(0, velocity), time=current_tick - last_tick))
            if velocity >= 0:
                raise e
        last_tick = current_tick



def save_midi(path, pitches, intervals, velocities, insts=None):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    if isinstance(pitches, list):
        for p, i, v, ins in zip(pitches, intervals, velocities, insts):
            # print('ins', ins)
            append_track_multi(file, p, i, v, ins)
    else:
        append_track(file, pitches, intervals, velocities)
    # track = MidiTrack()
    # file.tracks.append(track)
    # ticks_per_second = file.ticks_per_beat * 2.0
    #
    # events = []
    # for i in range(len(pitches)):
    #     events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
    #     events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    # events.sort(key=lambda row: row['time'])
    #
    # last_tick = 0
    # for event in events:
    #     current_tick = int(event['time'] * ticks_per_second)
    #     velocity = int(event['velocity'] * 127)
    #     if velocity > 127:
    #         velocity = 127
    #     pitch = int(round(hz_to_midi(event['pitch'])))
    #     track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
    #     last_tick = current_tick
    # print('saving to', path)
    file.save(path)


if __name__ == '__main__':

    def process(input_file, output_file):
        midi_data = parse_midi(input_file)
        np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith('.mid'):
                output_file = input_file[:-4] + '.tsv'
            elif input_file.endswith('.midi'):
                output_file = input_file[:-5] + '.tsv'
            else:
                print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
                continue

            yield (input_file, output_file)

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files())
