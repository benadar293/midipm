import os
from glob import glob
# from midi2tsv_musicnet_multi import parse_midi_multi
from eval_set_insts import *
from copy import deepcopy
import random
import torch

different_id_performances = []
different_id_performances += ['Bach Trio Sonatas', 'Mass in B Minor',
                              'Bach Concerto for 2 Harpsichords', 'Bach Concerto for 3 Harpsichords', 'Bach Organ',
                              'Sibelius Karelia Alla Marcia', 'Sibelius Karelia Intermezzo',
                              'Mozart Sonata 10'
                              ]
different_id_performances += ['Albeniz Suite Espagnole 6 Hojas', 'Albeniz Suite Espagnole',
                                'Tarrega A', 'Tarrega B',
                                'Sor Studies',
                                'Sor Mozart Variations']
different_id_performances += ['Sor Opus ' + str(i) for i in [6, 29, 31, 35]]

different_ids_reverse = {}
for diff_id in different_id_performances:
    for did in os.listdir('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/{}#0'.format(diff_id)):
        different_ids_reverse[did.split('#')[0]] = diff_id
print('reverse diff id', different_ids_reverse)

symphonies = [
            'Beethoven Symphony 1 V1', 'Beethoven Symphony 1 V2',
            'Beethoven Symphony 4',
            'Beethoven Symphony 8',
            'Beethoven Symphony 9 V1', 'Beethoven Symphony 9 V2'
        ]
symphonies += ['Mozart Symphony {}'.format(i) for i in [35, 36, 38, 41]]
symphonies += ['Brahms_SymphonyNo.2inDMajor', 'Brahms_SymphonyNo.3inFMajor']
symphonies += ['Beethoven_CoriolanOverture', 'Beethoven_EgmontOvertureOp.84', 'Beethoven_SymphonyNo.3Eroica']
symphonies += ['Brahms Haydn Variations']
symphonies += [
        'Mendelssohn_ScottishSymphony',
        'Brahms_TragicOverture',
        'Brahms Academic Festival Overture 1', 'Brahms Academic Festival Overture 2',
        'Mozart_MagicFluteOverture', 'Mozart_MarriageOfFigaro',
        'Sibelius Karelia Alla Marcia', 'Sibelius Karelia Intermezzo', 'Sibelius Valse Triste'
    ]

# choral = ['part' + '{}'.format(i).zfill(2) for i in set(range(1, 24)) - {3, 11, 15}]
choral = ['Mass in B Minor']


piano_concertos = ['Beethoven Piano Concertos',
                   ]
piano_concertos += ['Mozart Concerto {}'.format(i) for i in [15, 17, 19]]
piano_concertos += ['Chopin Piano Concerto 1', 'Chopin Piano Concerto 2']
piano_concertos += ['The Carnival of the Animals']


organ = ['Bach Trio Sonatas',
         ]
organ += ['Bach Organ']

cello = ['Bach Cello Suite no ' + str(i) for i in range(2, 7)]
violin = ['BWV ' + str(i) for i in range(1002, 1006)]

guitar = ['Albeniz Suite Espagnole 6 Hojas', 'Albeniz Suite Espagnole',
                    'Tarrega A', 'Tarrega B',
                    'Sor Studies',
                    'Sor Mozart Variations']
guitar += ['Sor Opus ' + str(i) for i in [6, 29, 31, 35]]

harpsichord = ['Rousset Goldberg', 'Bach WTC 2', 'English Suite 1']
harpsichord += ['Bach WTC 1 {}'.format(k) for k in ['A', 'E', 'F', 'G']]

harpsichord_violin = ['Bach Sonatas for Violin and Harpsichord B', 'Bach Sonatas for Violin and Harpsichord']
piano_trio = ['Beethoven Piano Trios']
piano_violin = ['Beethoven Violin Sonatas']
wind = ['MusicNet']
flute_harpsichord = ['Bach Sonata for Flute and Harpsichord BWV 1030', 'Bach Sonata for Flute and Harpsichord BWV 1031']

baroque_orchestra = ['Orchestral Suite {}'.format(i) for i in [1, 2, 3]]
baroque_orchestra += ['Bach Brandenburg Concerto 1 A', 'Bach Brandenburg Concerto 1 B']

large_orchestra = ['Tchaikovsky Nutcracker Suite']
large_orchestra += ['The Swan Lake Act {}'.format(i) for i in [1, 2, 3, 4]]

piano = ['Mozart Sonata 10', 'Mozart Sonata 13', 'Mozart Sonata 14', 'Mozart Sonata 19',
                    'Schubert_SonataInAMajorD.664', 'Schubert_SonataInAMinorD.784', 'Schubert_SonataInAMinorD.845',
                    'Schubert_SonataInAMinorD.959', 'Schubert_SonataInDMajorD.850', 'Schubert_SonataInEFlatMajorD.568']

def get_conversion(performance):
    if performance in different_ids_reverse:
        print('using reverse from', performance)
        performance = different_ids_reverse[performance]
        print('to', performance)
    res = deepcopy(conversion_map)
    if performance in piano_trio:
        res.update({6: 0, 48: 40, 41: 42})
    elif performance in piano_concertos:
        res.update({6: 0})
    elif performance in harpsichord:
        res.update({i: 6 for i in range(128)})
    elif performance in guitar:
        res.update({i: 24 for i in range(128)})
    elif performance in organ:
        res.update({i: 19 for i in range(128)})
    elif performance in piano:
        res.update({i: 0 for i in range(128)})
    elif performance in violin:
        res.update({i: 40 for i in range(128)})
    elif performance in cello:
        res.update({i: 42 for i in range(128)})
    elif performance in harpsichord_violin:
        res.update({41: 40, 42: 40, 48: 40, 24: 6, 0: 6})
    elif performance in piano_violin:
        res.update({41: 40, 42: 40, 48: 40, 24: 0, 6: 0})
    elif performance in large_orchestra:
        pass
        # res.update({24: 46})
    elif performance in flute_harpsichord:
        res.update({24: 6, 0: 6})
    # else:
    #     raise ValueError('conversion error', performance)
    return res

saved_ids = torch.load('/home/dcor/benmaman/PerformanceNet-master/runs/mel_large_320-230702-025658 DONE/ids.pt')['ids']
def apply_different_ids(performances):
    res = []
    for elem in performances:
        if elem in different_id_performances:
            curr_perfs = glob('/home/dcor/benmaman/PerformanceNet-master/Museopen16_flac/' + elem + '#0/*.flac')
            curr_perfs = [perf.split('/')[-1].split('#')[0] for perf in curr_perfs]
            assert all([cp in saved_ids for cp in curr_perfs])
            res.extend(curr_perfs)
            print('added different ids', curr_perfs)
        else:
            res.append(elem)
    return res



def get_eval_performances(f):
    if any([el in f for el in ['1079-', '1080-c01', 'mozart_string_quartet']]):
        return piano_trio
    elif '1080-c12' in f:
        return wind
    elif any([el in f for el in ['bach_invention_8_bwv_779', 'bwv808a', 'cou1', 'gig1']]):
        return harpsichord
    elif 'Bach_Suite_no1_BWV996_bourree' in f:
        return guitar
    elif 'Bach_Toccata_and_Fugue' in f:
        return organ
    elif any([el in f for el in ['beethoven_symphony_5', 'symphony_550', 'symphony_6']]):
        return symphonies
    elif any([el in f for el in ['bjsbmm11', 'monteverdi_libri_dei_madrigali', 'messiah_hallelujah_ORCH']]):
        return choral
    elif any([el in f for el in ['bwv29sin']]):
        return choral, baroque_orchestra
    elif any(['bwv{}'.format(el) in f for el in range(1027, 1030)]):
        return piano_trio
    elif 'concerto' in f:
        return piano_concertos
    elif 'cs1' in f:
        return cello
    elif 'Hungarian_Dance_5_Brahms' in f:
        return piano_trio, piano_violin
    elif any([el in f for el in ['Jobim_A_Felicidade', 'pfa-3alg']]):
        return guitar
    elif 'Jobim_Felicidade' in f:
        return flute_harpsichord
    elif 'kinokio_HN' in f:
        return wind
    elif any([el in f for el in ['vp3', 'paganini_capriccio_24']]):
        return violin

    elif any([el in f for el in ['mahler_symphony_1_2', 'p1', 'p3',
                                 'Rodrigo_Concierto_de_Aranjuez_1_Allegro_con_Spirito',
                                 'Rodrigo_Concierto_de_Aranjuez_3_Allegro_Gentile'
                                 ]]):
        return large_orchestra
    elif 'orch4' in f:
        return baroque_orchestra

    elif 'suite_bergamasque_1_(c)galimberti' in f or 'sonate_01_' in f:
        return piano
    elif 'Vivaldi_Concerto' in f:
        return harpsichord_violin
    else:
        print('not found', f)
        raise ValueError







# src = '/home/dcor/benmaman/PerformanceNet-master/EvalSet'

# midis_and_performances = []
# max_perfs = 3

# for f in glob(src + '/*.mid'):
#     print(f)
#     # midi = parse_midi_multi(f, force_instrument=None)
#     # print('input instruments:', set(midi[:, -1].astype(int)))
#     curr_eval_performances = get_eval_performances(f)
#     if isinstance(curr_eval_performances, list):
#         curr_performances = apply_different_ids(curr_eval_performances)
#         if len(curr_performances) > max_perfs:
#             curr_performances = random.sample(curr_performances, k=max_perfs)
#         for performance in curr_performances:
#             midis_and_performances.append((f, performance))
#     else:
#         print('not list', curr_eval_performances)
#         assert isinstance(curr_eval_performances, tuple)
#         for curr_performances in curr_eval_performances:
#             print('here curr', curr_performances)
#             curr_performances = apply_different_ids(curr_performances)
#             print('not list different', curr_performances)
#             if len(curr_performances) > max_perfs:
#                 curr_performances = random.sample(curr_performances, k=max_perfs)
#             for performance in curr_performances:
#                 midis_and_performances.append((f, performance))
# print('Eval midis and performances:')
# for el in midis_and_performances:
#     print(el, ',', sep='')
# print('all:', len(midis_and_performances))