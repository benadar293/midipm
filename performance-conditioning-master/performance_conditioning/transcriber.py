"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""
import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from torch.nn import TransformerEncoderLayer, TransformerEncoder
# from .mel import melspectrogram
# from onsets_and_frames.lstm import BiLSTM
# from onsets_and_frames.mel import melspectrogram
from onsets_and_frames.constants import *

from torch.nn import DataParallel

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, onset_complexity=1):
        super().__init__()
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        print('onset complexity:', onset_complexity)
        transformer_sequence_model = lambda output_size, n_layers, n_heads: \
            TransformerEncoder(TransformerEncoderLayer(d_model=output_size, nhead=n_heads),
                               num_layers=n_layers)


        # self.onset_stack = nn.Sequential(
        #     ConvStack(input_features, model_size),
        #     sequence_model(model_size, model_size),
        #     nn.Linear(model_size, output_features),
        #     nn.Sigmoid()
        # )
        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            # nn.Linear(model_size, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            # sequence_model(output_features * 3 * len(INSTRUMENT_MAPPING),
            #                output_features * 3 * len(INSTRUMENT_MAPPING)),
            # nn.Linear(output_features * 3 * len(INSTRUMENT_MAPPING), output_features),
            # sequence_model(output_features * 3, output_features * 3),
            # nn.Linear(output_features * 3, output_features),

            ## sequence_model(output_features * 3, model_size),
            # sequence_model(88 * 3, model_size),
            # nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        # self.combined_stack = nn.Sequential(
        #     sequence_model(output_features * 3, 48 * 16),
        #     nn.Linear(48 * 16, output_features),
        #     nn.Sigmoid()
        # )
        # self.combined_stack = nn.Sequential(
        #     nn.Linear(output_features * 3, model_size),
        #     transformer_sequence_model(model_size, 1, 8),
        #     nn.Linear(model_size, output_features),
        #     nn.Sigmoid()
        # )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )


    # def forward(self, mel):
    #     onset_pred = self.onset_stack(mel)
    #     offset_pred = self.offset_stack(mel)
    #     activation_pred = self.frame_stack(mel)
    #
    #     onset_detached = onset_pred.detach()
    #     offset_detached = offset_pred.detach()
    #     n_copies = len(INSTRUMENT_MAPPING)
    #     combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
    #     # combined_pred = torch.cat(n_copies * [onset_detached] + n_copies * [offset_detached] + n_copies * [activation_pred], dim=-1)
    #     frame_pred = self.combined_stack(combined_pred)
    #     velocity_pred = self.velocity_stack(mel)
    #     return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)
        # onset_detached = onset_detached[..., -1]

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        # velocity_mask = batch['velocity_mask'].squeeze()

        # if onset_label.shape[-1] < MAX_MIDI - MIN_MIDI + 1:
        #     t = onset_label.shape[0]
        #     p = MAX_MIDI - MIN_MIDI + 1 - onset_label.shape[-1]
        #     onset_label = torch.cat((onset_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
        #     offset_label = torch.cat((offset_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
        #     frame_label = torch.cat((frame_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
        #     velocity_label = torch.cat((velocity_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)


        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            # b, t, p = frame_pred.reshape
            # frame_pred_reshaped = frame_pred.reshape((b, t, p // N_KEYS, N_KEYS))
            # frame_pred, _ = frame_pred_reshaped.max(dim=2)
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        if 'onsets_mask' not in batch:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        else:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

                # l1 = F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none')
            # l1 = l1[..., 1: -1, :]
            # l2 = F.binary_cross_entropy(predictions['onset'][..., : -1, :], onset_label[..., 1:, :], reduction='none')
            # l2 = l2[..., : -1, :]
            # l3 = F.binary_cross_entropy(predictions['onset'][..., 1:, :], onset_label[..., : -1, :], reduction='none')
            # l3 = l3[..., 1:, :]
            # min_loss = torch.min(l1, l2)
            # # min_loss = torch.min(min_loss, l3)
            # losses['loss/onset'] = min_loss
            # onset_mask = batch['onsets_mask'][..., 1: -1, :]



            onset_mask = batch['onsets_mask']
            frame_mask = batch['frames_mask']
            offset_mask = batch['offsets_mask']

            # onset_mask = len(INSTRUMENT_MAPPING) * onset_label + (1 - onset_label)
            # frame_mask = len(INSTRUMENT_MAPPING) * frame_label + (1 - frame_label)
            # offset_mask = len(INSTRUMENT_MAPPING) * offset_label + (1 - offset_label)
            # print('onset shapes', onset_label.shape, onset_pred.shape, len(INSTRUMENT_MAPPING) * onset_label.mean())
            # print('frame shapes', frame_label.shape, frame_pred.shape, frame_label.mean())
            # print('offset shapes', offset_label.shape, offset_pred.shape, offset_label.mean())



            for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
                losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

                # losses['loss/' + loss_key] = (mask * losses_unreduced['loss/' + loss_key]).mean()

                # losses['loss/' + loss_key] = (predictions[loss_key] * losses['loss/' + loss_key]).mean()
            # del losses_unreduced['loss/' + loss_key]
            # torch.cuda.empty_cache()

        # losses = {}
        # losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # for name, lab in zip(['onset', 'offset', 'frame'], [onset_label, offset_label, frame_label]):
        #     l1 = F.binary_cross_entropy(predictions[name], lab, reduction='none')
        #     l1 = l1[..., 1: -1, :]
        #     l2 = F.binary_cross_entropy(predictions[name][..., : -1, :], lab[..., 1:, :], reduction='none')
        #     l2 = l2[..., : -1, :]
        #     l3 = F.binary_cross_entropy(predictions[name][..., 1:, :], lab[..., : -1, :], reduction='none')
        #     l3 = l3[..., 1:, :]
        #     min_loss = torch.min(l1, l2)
        #     min_loss = torch.min(min_loss, l3)
        #     losses['loss/' + name] = min_loss.mean()


        # if self.n_instruments > 1:
        #     predictions_max = {k: v.max(dim=-1)[0] for k, v in predictions.items()}
        #     # for elem in [onset_label, offset_label, frame_label, velocity_label]:
        #     #     print('elem shape', elem.shape)
        #     losses_max = {
        #         'loss_max/onset': F.binary_cross_entropy(predictions_max['onset'], onset_label.max(dim=-1)[0]),
        #         'loss_max/offset': F.binary_cross_entropy(predictions_max['offset'], offset_label.max(dim=-1)[0]),
        #         'loss_max/frame': F.binary_cross_entropy(predictions_max['frame'], frame_label.max(dim=-1)[0]),
        #         'loss_max/velocity': self.velocity_loss(predictions_max['velocity'],
        #                                             velocity_label.max(dim=-1)[0], onset_label.max(dim=-1)[0])
        #     }
        #     losses_max.update({k: self.loss_weight * v for k, v in losses.items()})
        #     losses = losses_max

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

    # def velocity_loss(self, velocity_pred, velocity_label, onset_label, mask=None):
    #     to_sum = onset_label
    #     if mask is not None:
    #         to_sum = to_sum[mask]
    #     denominator = to_sum.sum()
    #     if denominator.item() == 0:
    #         return denominator
    #     else:
    #         res = (onset_label * (velocity_label - velocity_pred) ** 2)
    #         if mask is not None:
    #             res = res[mask]
    #         return res.sum() / denominator


#
# class OnsetsAndFrames(nn.Module):
#     def __init__(self, input_features, output_features, model_complexity=48, onset_complexity=1):
#         super().__init__()
#         model_size = model_complexity * 16
#         sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
#         print('onset complexity:', onset_complexity)
#
#         onset_model_size = int(onset_complexity * model_size)
#         self.onset_stack = nn.Sequential(
#             ConvStack(input_features, onset_model_size),
#             # transformer_sequence_model(model_size, 1, 8),
#             sequence_model(onset_model_size, onset_model_size),
#             nn.Linear(onset_model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.offset_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             # transformer_sequence_model(model_size, 1, 8),
#             sequence_model(model_size, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.frame_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             nn.Linear(model_size, output_features),
#             # nn.Linear(model_size, 88),
#             nn.Sigmoid()
#         )
#         self.combined_stack = nn.Sequential(
#             sequence_model(output_features * 3, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#
#         self.velocity_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             nn.Linear(model_size, output_features)
#         )
#
#
#     def forward(self, mel):
#         onset_pred = self.onset_stack(mel)
#         offset_pred = self.offset_stack(mel)
#         activation_pred = self.frame_stack(mel)
#
#         combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
#         frame_pred = self.combined_stack(combined_pred)
#         velocity_pred = self.velocity_stack(mel)
#         return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
#
#     def run_on_batch(self, batch, parallel_model=None, multi=False, already_mel=False):
#         audio_label = batch['audio']
#
#         onset_label = batch['onset']
#         offset_label = batch['offset']
#         frame_label = batch['frame']
#         if 'velocity' in batch:
#             velocity_label = batch['velocity']
#
#         if already_mel:
#             mel = audio_label
#         else:
#             mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
#
#         if not parallel_model:
#             onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
#         else:
#             onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)
#
#         if multi:
#             onset_pred = onset_pred[..., : N_KEYS]
#             offset_pred = offset_pred[..., : N_KEYS]
#             # b, t, p = frame_pred.reshape
#             # frame_pred_reshaped = frame_pred.reshape((b, t, p // N_KEYS, N_KEYS))
#             # frame_pred, _ = frame_pred_reshaped.max(dim=2)
#             frame_pred = frame_pred[..., : N_KEYS]
#             velocity_pred = velocity_pred[..., : N_KEYS]
#         predictions = {
#             'onset': onset_pred.reshape(*onset_label.shape),
#             'offset': offset_pred.reshape(*offset_label.shape),
#             'frame': frame_pred.reshape(*frame_label.shape),
#             # 'velocity': velocity_pred.reshape(*velocity_label.shape)
#         }
#         if 'velocity' in batch:
#             predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)
#
#         if 'onsets_mask' not in batch:
#             losses = {
#                 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
#                 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
#                 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
#                 # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#             }
#             if 'velocity' in batch:
#                 losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#
#         else:
#             losses = {
#                 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
#                 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
#                 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
#                 # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#             }
#             if 'velocity' in batch:
#                 losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#
#             onset_mask = batch['onsets_mask']
#             frame_mask = batch['frames_mask']
#             offset_mask = batch['offsets_mask']
#
#             for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
#                 losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
#
#         return predictions, losses
#
#     def velocity_loss(self, velocity_pred, velocity_label, onset_label):
#         denominator = onset_label.sum()
#         if denominator.item() == 0:
#             return denominator
#         else:
#             return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

# class OnsetsAndFrames(nn.Module):
#     def __init__(self, input_features, output_features, model_complexity=48, onset_complexity=1):
#         super().__init__()
#         model_size = model_complexity * 16
#         sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
#         print('onset complexity:', onset_complexity)
#         transformer_sequence_model = lambda output_size, n_layers, n_heads: \
#             TransformerEncoder(TransformerEncoderLayer(d_model=output_size, nhead=n_heads),
#                                num_layers=n_layers)
#
#
#         # self.onset_stack = nn.Sequential(
#         #     ConvStack(input_features, model_size),
#         #     sequence_model(model_size, model_size),
#         #     nn.Linear(model_size, output_features),
#         #     nn.Sigmoid()
#         # )
#         onset_model_size = int(onset_complexity * model_size)
#         self.onset_stack = nn.Sequential(
#             ConvStack(input_features, onset_model_size),
#             # transformer_sequence_model(model_size, 1, 8),
#             sequence_model(onset_model_size, onset_model_size),
#             nn.Linear(onset_model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.offset_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             # transformer_sequence_model(model_size, 1, 8),
#             sequence_model(model_size, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.frame_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             nn.Linear(model_size, output_features),
#             # nn.Linear(model_size, 88),
#             nn.Sigmoid()
#         )
#         self.combined_stack = nn.Sequential(
#             sequence_model(output_features * 3, model_size),
#             nn.Linear(model_size, output_features),
#             # sequence_model(output_features * 3 * len(INSTRUMENT_MAPPING),
#             #                output_features * 3 * len(INSTRUMENT_MAPPING)),
#             # nn.Linear(output_features * 3 * len(INSTRUMENT_MAPPING), output_features),
#             # sequence_model(output_features * 3, output_features * 3),
#             # nn.Linear(output_features * 3, output_features),
#
#             ## sequence_model(output_features * 3, model_size),
#             # sequence_model(88 * 3, model_size),
#             # nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         # self.combined_stack = nn.Sequential(
#         #     sequence_model(output_features * 3, 48 * 16),
#         #     nn.Linear(48 * 16, output_features),
#         #     nn.Sigmoid()
#         # )
#         # self.combined_stack = nn.Sequential(
#         #     nn.Linear(output_features * 3, model_size),
#         #     transformer_sequence_model(model_size, 1, 8),
#         #     nn.Linear(model_size, output_features),
#         #     nn.Sigmoid()
#         # )
#         self.velocity_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             nn.Linear(model_size, output_features)
#         )
#
#
#     def forward(self, mel):
#         onset_pred = self.onset_stack(mel)
#         offset_pred = self.offset_stack(mel)
#         activation_pred = self.frame_stack(mel)
#
#         onset_detached = onset_pred.detach()
#         offset_detached = offset_pred.detach()
#         n_copies = len(INSTRUMENT_MAPPING)
#         combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
#         # combined_pred = torch.cat(n_copies * [onset_detached] + n_copies * [offset_detached] + n_copies * [activation_pred], dim=-1)
#         frame_pred = self.combined_stack(combined_pred)
#         velocity_pred = self.velocity_stack(mel)
#         return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
#
#     def run_on_batch(self, batch, parallel_model=None, multi=False):
#         audio_label = batch['audio']
#
#         onset_label = batch['onset']
#         offset_label = batch['offset']
#         frame_label = batch['frame']
#         if 'velocity' in batch:
#             velocity_label = batch['velocity']
#         # velocity_mask = batch['velocity_mask'].squeeze()
#
#         # if onset_label.shape[-1] < MAX_MIDI - MIN_MIDI + 1:
#         #     t = onset_label.shape[0]
#         #     p = MAX_MIDI - MIN_MIDI + 1 - onset_label.shape[-1]
#         #     onset_label = torch.cat((onset_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
#         #     offset_label = torch.cat((offset_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
#         #     frame_label = torch.cat((frame_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
#         #     velocity_label = torch.cat((velocity_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
#
#
#         mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
#         if not parallel_model:
#             onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
#         else:
#             onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)
#
#         if multi:
#             onset_pred = onset_pred[..., : N_KEYS]
#             offset_pred = offset_pred[..., : N_KEYS]
#             # b, t, p = frame_pred.reshape
#             # frame_pred_reshaped = frame_pred.reshape((b, t, p // N_KEYS, N_KEYS))
#             # frame_pred, _ = frame_pred_reshaped.max(dim=2)
#             frame_pred = frame_pred[..., : N_KEYS]
#             velocity_pred = velocity_pred[..., : N_KEYS]
#         predictions = {
#             'onset': onset_pred.reshape(*onset_label.shape),
#             'offset': offset_pred.reshape(*offset_label.shape),
#             'frame': frame_pred.reshape(*frame_label.shape),
#             # 'velocity': velocity_pred.reshape(*velocity_label.shape)
#         }
#         if 'velocity' in batch:
#             predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)
#
#         if 'onsets_mask' not in batch:
#             losses = {
#                 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
#                 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
#                 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
#                 # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#             }
#             if 'velocity' in batch:
#                 losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#
#         else:
#             losses = {
#                 'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
#                 'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
#                 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
#                 # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#             }
#             if 'velocity' in batch:
#                 losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#
#                 # l1 = F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none')
#             # l1 = l1[..., 1: -1, :]
#             # l2 = F.binary_cross_entropy(predictions['onset'][..., : -1, :], onset_label[..., 1:, :], reduction='none')
#             # l2 = l2[..., : -1, :]
#             # l3 = F.binary_cross_entropy(predictions['onset'][..., 1:, :], onset_label[..., : -1, :], reduction='none')
#             # l3 = l3[..., 1:, :]
#             # min_loss = torch.min(l1, l2)
#             # # min_loss = torch.min(min_loss, l3)
#             # losses['loss/onset'] = min_loss
#             # onset_mask = batch['onsets_mask'][..., 1: -1, :]
#
#
#
#             onset_mask = batch['onsets_mask']
#             frame_mask = batch['frames_mask']
#             offset_mask = batch['offsets_mask']
#
#             # onset_mask = len(INSTRUMENT_MAPPING) * onset_label + (1 - onset_label)
#             # frame_mask = len(INSTRUMENT_MAPPING) * frame_label + (1 - frame_label)
#             # offset_mask = len(INSTRUMENT_MAPPING) * offset_label + (1 - offset_label)
#             # print('onset shapes', onset_label.shape, onset_pred.shape, len(INSTRUMENT_MAPPING) * onset_label.mean())
#             # print('frame shapes', frame_label.shape, frame_pred.shape, frame_label.mean())
#             # print('offset shapes', offset_label.shape, offset_pred.shape, offset_label.mean())
#
#
#
#             for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
#                 losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
#
#                 # losses['loss/' + loss_key] = (mask * losses_unreduced['loss/' + loss_key]).mean()
#
#                 # losses['loss/' + loss_key] = (predictions[loss_key] * losses['loss/' + loss_key]).mean()
#             # del losses_unreduced['loss/' + loss_key]
#             # torch.cuda.empty_cache()
#
#         # losses = {}
#         # losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
#         # for name, lab in zip(['onset', 'offset', 'frame'], [onset_label, offset_label, frame_label]):
#         #     l1 = F.binary_cross_entropy(predictions[name], lab, reduction='none')
#         #     l1 = l1[..., 1: -1, :]
#         #     l2 = F.binary_cross_entropy(predictions[name][..., : -1, :], lab[..., 1:, :], reduction='none')
#         #     l2 = l2[..., : -1, :]
#         #     l3 = F.binary_cross_entropy(predictions[name][..., 1:, :], lab[..., : -1, :], reduction='none')
#         #     l3 = l3[..., 1:, :]
#         #     min_loss = torch.min(l1, l2)
#         #     min_loss = torch.min(min_loss, l3)
#         #     losses['loss/' + name] = min_loss.mean()
#
#
#         # if self.n_instruments > 1:
#         #     predictions_max = {k: v.max(dim=-1)[0] for k, v in predictions.items()}
#         #     # for elem in [onset_label, offset_label, frame_label, velocity_label]:
#         #     #     print('elem shape', elem.shape)
#         #     losses_max = {
#         #         'loss_max/onset': F.binary_cross_entropy(predictions_max['onset'], onset_label.max(dim=-1)[0]),
#         #         'loss_max/offset': F.binary_cross_entropy(predictions_max['offset'], offset_label.max(dim=-1)[0]),
#         #         'loss_max/frame': F.binary_cross_entropy(predictions_max['frame'], frame_label.max(dim=-1)[0]),
#         #         'loss_max/velocity': self.velocity_loss(predictions_max['velocity'],
#         #                                             velocity_label.max(dim=-1)[0], onset_label.max(dim=-1)[0])
#         #     }
#         #     losses_max.update({k: self.loss_weight * v for k, v in losses.items()})
#         #     losses = losses_max
#
#         return predictions, losses
#
#     def velocity_loss(self, velocity_pred, velocity_label, onset_label):
#         denominator = onset_label.sum()
#         if denominator.item() == 0:
#             return denominator
#         else:
#             return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
#
#     # def velocity_loss(self, velocity_pred, velocity_label, onset_label, mask=None):
#     #     to_sum = onset_label
#     #     if mask is not None:
#     #         to_sum = to_sum[mask]
#     #     denominator = to_sum.sum()
#     #     if denominator.item() == 0:
#     #         return denominator
#     #     else:
#     #         res = (onset_label * (velocity_label - velocity_pred) ** 2)
#     #         if mask is not None:
#     #             res = res[mask]
#     #         return res.sum() / denominator


class OnsetsAndFramesMulti(OnsetsAndFrames):
    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (keys, shape[-1] // keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-1)

        offset_detached = offset_pred.detach()
        offset_detached = offset_detached.reshape(new_shape)
        offset_detached, _ = offset_detached.max(axis=-1)


        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class OnsetsAndFramesMultiV2(OnsetsAndFrames):
    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class OnsetsAndFramesMultiV3(OnsetsAndFrames):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=8):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        transformer_sequence_model = lambda output_size, n_layers, n_heads: \
            TransformerEncoder(TransformerEncoderLayer(d_model=output_size, nhead=n_heads),
                               num_layers=n_layers)

        onset_model_size = int(onset_complexity * model_size)
        print('onset model size', onset_model_size)
        print('onset output size', output_features * n_instruments)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )
        # self.onset_stack = nn.Sequential(
        #     ConvStack(input_features, model_size),
        #     # transformer_sequence_model(model_size, 1, 8),
        #     sequence_model(model_size, model_size),
        #     nn.Linear(model_size, output_features * n_instruments),
        #     nn.Sigmoid()
        # )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            # nn.Linear(model_size, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            # sequence_model(output_features * 3, output_features * 3),
            # nn.Linear(output_features * 3, output_features),
            nn.Sigmoid()
        )

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)
        # onset_detached = onset_detached[..., -1]


        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred#, velocity_pred

class OnsetsAndFramesMultiV4(OnsetsAndFramesMultiV3):
    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]

            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        if 'velocity' in batch:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        return predictions, losses


def compress_tensor_across_octave(notes, apply_min=False):
    keys = (MAX_MIDI - MIN_MIDI + 1)
    if len(notes.shape) == 2:
        notes = notes.unsqueeze(0)
    b, time, instruments = notes.shape[0], notes.shape[1], notes.shape[2] // keys
    octaves = 8
    notes = notes.reshape((b, time, instruments, keys))
    padding = torch.zeros((b, time, instruments, 8), dtype=notes.dtype).cuda()
    notes = torch.cat((notes, padding), dim=-1)
    notes = notes.reshape((b, time, instruments, octaves, 12))
    res, _ = notes.max(dim=3) if not apply_min else notes.min(dim=3)
    return res.reshape(b, time, 12 * instruments)

def compress_tensor_across_instrument(notes, apply_min=False):
    if len(notes.shape) == 2:
        notes = notes.unsqueeze(0)
    keys = (MAX_MIDI - MIN_MIDI + 1)
    b, time, instruments = notes.shape[0], notes.shape[1], notes.shape[2] // keys
    notes = notes.reshape((b, time, instruments, keys))
    res, _ = notes.max(dim=2) if not apply_min else notes.min(dim=2)
    return res

def compress_tensor_across_pitch(notes, apply_min=False):
    if len(notes.shape) == 2:
        notes = notes.unsqueeze(0)
    keys = (MAX_MIDI - MIN_MIDI + 1)
    b, time, instruments = notes.shape[0], notes.shape[1], notes.shape[2] // keys
    notes = notes.reshape((b, time, instruments, keys))
    res, _ = notes.max(dim=-1) if not apply_min else notes.min(dim=-1)
    return res

def compress_tensor_across_octave_and_instrument(notes, apply_min=False):
    if len(notes.shape) == 2:
        notes = notes.unsqueeze(0)
    keys = (MAX_MIDI - MIN_MIDI + 1)
    b, time, instruments = notes.shape[0], notes.shape[1], notes.shape[2] // keys
    octaves = 8
    notes = notes.reshape((b, time, instruments, keys))
    padding = torch.zeros((b, time, instruments, 8), dtype=notes.dtype).cuda()
    notes = torch.cat((notes, padding), dim=-1)
    notes = notes.reshape((b, time, instruments, octaves, 12))
    res = notes.amax(dim=(2, 3)) if not apply_min else notes.amin(dim=(2, 3))
    return res.reshape(b, time, 12)


class OnsetsAndFramesMultiV10(OnsetsAndFramesMultiV3):
    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        if 'onsets_mask' not in batch:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        else:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)


            onset_mask = batch['onsets_mask']
            frame_mask = batch['frames_mask']
            offset_mask = batch['offsets_mask']

            for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
                losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

            octave_inv_pred = compress_tensor_across_octave(predictions['onset'])
            octave_inv_label = compress_tensor_across_octave(onset_label)
            octave_inv_mask = compress_tensor_across_octave(onset_mask)
            octave_inv_loss = F.binary_cross_entropy(octave_inv_pred, octave_inv_label, reduction='none')
            losses['loss/onset'] += (octave_inv_mask * octave_inv_loss).mean()

            octave_inv_pred_frame = compress_tensor_across_octave(predictions['frame'])
            octave_inv_label_frame = compress_tensor_across_octave(frame_label)
            octave_inv_mask_frame = compress_tensor_across_octave(frame_mask)
            octave_inv_loss_frame = F.binary_cross_entropy(octave_inv_pred_frame, octave_inv_label_frame, reduction='none')
            losses['loss/frame'] += (octave_inv_mask_frame * octave_inv_loss_frame).mean()

            octave_inv_pred = compress_tensor_across_octave_and_instrument(predictions['onset'])
            octave_inv_label = compress_tensor_across_octave_and_instrument(onset_label)
            octave_inv_mask = compress_tensor_across_octave_and_instrument(onset_mask)
            octave_inv_loss = F.binary_cross_entropy(octave_inv_pred, octave_inv_label, reduction='none')
            losses['loss/onset'] += (octave_inv_mask * octave_inv_loss).mean()

            octave_inv_pred_frame = compress_tensor_across_octave_and_instrument(predictions['frame'])
            octave_inv_label_frame = compress_tensor_across_octave_and_instrument(frame_label)
            octave_inv_mask_frame = compress_tensor_across_octave_and_instrument(frame_mask)
            octave_inv_loss_frame = F.binary_cross_entropy(octave_inv_pred_frame, octave_inv_label_frame,
                                                           reduction='none')
            losses['loss/frame'] += (octave_inv_mask_frame * octave_inv_loss_frame).mean()

        return predictions, losses



class OnsetsAndFramesMultiV12(OnsetsAndFramesMultiV3):
    def run_on_batch(self, batch, melspectrogram, parallel_model=None, multi=False, positive_weight=10., inv_positive_weight=5.,
                     label_dropout=0.5, apply_vel_loss=True):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        # print('run', onset_label.shape)
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        # print('run mel', mel.shape)

        # if not parallel_model:
        #     onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        # else:
        #     onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            # velocity_pred = velocity_pred[..., : N_KEYS]
        # print('shapes', onset_pred.shape, onset_label.shape)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        # losses = {
        #     'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        #     'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
        #     'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
        #     # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # }
        # if 'velocity' in batch:
        #     losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='mean'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='mean'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
        if 'velocity' in batch and apply_vel_loss:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        # inst_classes = 3.
        # positive_weight = (1 + inst_classes) / 2.
        # positive_weight *= 256 / HOP_LENGTH
        # positive_weight = inst_classes
        onset_mask = 1. * onset_label #+ 1.
        # positive_weight = 5.
        # positive_weight = 3.
        # positive_weight = 2.
        # positive_weight = 4.
        # positive_weight = 10. if HOP_LENGTH == 256 else 15.
        # positive_weight = 15. if HOP_LENGTH == 256 else 15.



        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        # if HOP_LENGTH == 128:
        #     onset_mask[..., -N_KEYS:] *= 3.
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        # onset_mask *= 2
        onset_mask += 1

        if label_dropout:
            # onset_mask *= (torch.randn(onset_mask.shape) >= 0).cuda()
            onset_mask *= (torch.rand(onset_mask.shape) >= label_dropout).cuda()


        # offset_mask = 1. * offset_label
        # offset_positive_weight = 10.
        # offset_mask *= (offset_positive_weight - 1)
        # offset_mask += 1.

        # frame_mask = 1. * frame_label
        # frame_positive_weight = 2.
        # frame_mask *= (frame_positive_weight - 1)
        # frame_mask += 1.


        # for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
        #     losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        for loss_key, mask in zip(['onset'], [onset_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        # for loss_key in ['offset', 'frame']:
        #     print(loss_key, 'loss shape', losses['loss/' + loss_key].shape)
        #     losses['loss/' + loss_key] = (losses['loss/' + loss_key]).mean()


        # else:
        #     losses = {
        #         'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
        #         'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
        #         'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
        #         # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        #     }
        #     if 'velocity' in batch:
        #         losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        #
        #     onset_mask = batch['onsets_mask']
        #     frame_mask = batch['frames_mask']
        #     offset_mask = batch['offsets_mask']
        #
        #     for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
        #         losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        #
        #     for pred, label, mask in zip(['onset', 'frame', 'offset'],
        #                                  [onset_label, frame_label, offset_label],
        #                                  [onset_mask, frame_mask, offset_mask]):
        #         if pred == 'onset':
        #             compress_functions = [compress_tensor_across_octave,
        #                                  compress_tensor_across_octave_and_instrument,
        #                                  compress_tensor_across_instrument
        #                                  ]
        #         else:
        #             compress_functions = [compress_tensor_across_instrument]
        #
        #         for comress_func in compress_functions:
        #             inv_pred = comress_func(predictions[pred])
        #             inv_label = comress_func(label)
        #             inv_mask = comress_func(mask, apply_min=True)
        #             # print('mask size', inv_mask.sum() / inv_mask.nelement())
        #             octave_inv_loss = F.binary_cross_entropy(inv_pred, inv_label, reduction='none')
        #             losses['loss/' + pred] += (inv_mask * octave_inv_loss).mean()

        return predictions, losses



class OnsetsAndFrames320(OnsetsAndFramesMultiV3):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=8):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )

        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            # nn.Linear(model_size, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            # sequence_model(output_features * 3, output_features * 3),
            # nn.Linear(output_features * 3, output_features),
            nn.Sigmoid()
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)
        # onset_detached = onset_detached[..., -1]

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch, mel_spec, parallel_model=None, positive_weight=10., inv_positive_weight=5.,
                     label_dropout=0.15, frame_dropout=0.5):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        # print('frame percent', frame_label.sum() / torch.numel(frame_label), 'shape', frame_label.shape)
        # print('frame percent 1', (frame_label == 1).sum() / torch.numel(frame_label), 'shape', frame_label.shape)
        # print('frame percent 0', (frame_label == 0).sum() / torch.numel(frame_label), 'shape', frame_label.shape)

        mel = mel_spec(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        # print('transcriber mel shape', mel.shape)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred = parallel_model(mel)

        # print('frame_pred percent', frame_pred.sum() / torch.numel(frame_pred), 'shape', frame_pred.shape)
        # print('frame_pred percent 1', (frame_pred == 1).sum() / torch.numel(frame_pred), 'shape', frame_pred.shape)
        # print('frame_pred percent 0', (frame_pred == 0).sum() / torch.numel(frame_pred), 'shape', frame_pred.shape)
        # print('frame shape', frame_pred.shape, frame_label.shape)
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
        }

        losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='mean'),
                # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='mean'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
            }

        onset_mask = 1. * onset_label #+ 1.
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1

        frame_mask = torch.ones_like(frame_label)

        if label_dropout:
            onset_mask *= (torch.rand(onset_mask.shape) >= label_dropout).cuda()
        if frame_dropout:
            frame_mask *= (torch.rand(frame_mask.shape) >= frame_dropout).cuda()

        for loss_key, mask in zip(['onset', 'frame'], [onset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        # for loss_key, mask in zip(['onset'], [onset_mask]):
        #     losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses



class OnsetsAndFramesMultiSCEM(OnsetsAndFramesMultiV3):
    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        if 'velocity' in batch:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        for pred, label in zip(['onset', 'frame', 'offset'],
                                     [onset_label, frame_label, offset_label]):
            if pred == 'onset':
                compress_functions = [compress_tensor_across_octave,
                                      compress_tensor_across_octave_and_instrument,
                                      compress_tensor_across_instrument,
                                      compress_tensor_across_pitch
                                      ]
            else:
                compress_functions = [compress_tensor_across_instrument]

            for comress_func in compress_functions:
                inv_pred = comress_func(predictions[pred])
                inv_label = comress_func(label)
                octave_inv_loss = F.binary_cross_entropy(inv_pred, inv_label)
                losses['loss/' + pred] += octave_inv_loss

        return predictions, losses



class OnsetsAndFramesMultiV13(OnsetsAndFramesMultiV12):
    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        if 'onsets_mask' not in batch:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        else:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

            onset_mask = batch['onsets_mask']

            for loss_key, mask in zip(['onset'], [onset_mask]):
                losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

            for pred, label, mask in zip(['onset', 'frame', 'offset'],
                                         [onset_label, frame_label, offset_label],
                                         [onset_mask, None, None]):
                if pred == 'onset':
                    compress_functions = [compress_tensor_across_octave,
                                         compress_tensor_across_octave_and_instrument,
                                         compress_tensor_across_instrument
                                         ]
                else:
                    compress_functions = [compress_tensor_across_instrument]

                for comress_func in compress_functions:
                    inv_pred = comress_func(predictions[pred])
                    inv_label = comress_func(label)
                    octave_inv_loss = F.binary_cross_entropy(inv_pred, inv_label, reduction='none')
                    if pred == 'onset':
                        inv_mask = comress_func(mask)
                        octave_inv_loss *= inv_mask
                    losses['loss/' + pred] += (octave_inv_loss).mean()

        return predictions, losses


class OnsetsAndFramesMultiFineTune(OnsetsAndFramesMultiV12):
    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        onset_pred = onset_pred[..., -N_KEYS:]
        velocity_pred = velocity_pred[..., -N_KEYS:]


        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        if 'onsets_mask' not in batch:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        else:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)


            # onset_mask = batch['onsets_mask']
            onset_mask = 1. * onset_label  # + 1.
            # onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
            # if HOP_LENGTH == 128:
            #     onset_mask[..., -N_KEYS:] *= 3.
            onset_mask[..., -N_KEYS:] *= 9
            # onset_mask *= 2
            onset_mask += 1

            frame_mask = batch['frames_mask']
            offset_mask = batch['offsets_mask']


            for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
                losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses

# oaf = OnsetsAndFrames(100, 100)
# oaf.onset_stack[1].flatten_parameters()
#
# print(oaf.onset_stack[1])


# class OnsetsFramesInstruments(nn.Module):
#     def __init__(self, input_features, output_features, model_complexity=48, n_instruments=1,
#                  loss_weight=None):
#         super().__init__()
#         self.n_instruments = n_instruments
#         self.loss_weight = loss_weight if loss_weight is not None else (1 / self.n_instruments)
#         model_size = model_complexity * 16
#         sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
#
#         output_features *= n_instruments
#
#         self.onset_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             sequence_model(model_size, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.offset_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             sequence_model(model_size, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.frame_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.combined_stack = nn.Sequential(
#             # sequence_model(output_features * 3 + INSTRUMENTS, model_size),
#             sequence_model(output_features * 3, model_size),
#             nn.Linear(model_size, output_features),
#             nn.Sigmoid()
#         )
#         self.velocity_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             nn.Linear(model_size, output_features)
#         )
#         self.instrument_stack = nn.Sequential(
#             ConvStack(input_features, model_size),
#             sequence_model(model_size, model_size),
#             nn.Linear(model_size, INSTRUMENTS),
#             nn.Sigmoid()
#         )
#
#     def forward(self, mel):
#         onset_pred = self.onset_stack(mel)
#         offset_pred = self.offset_stack(mel)
#         activation_pred = self.frame_stack(mel)
#         instrument_pred = self.instrument_stack(mel)
#         # combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(),
#         #                            activation_pred, instrument_pred.detach()], dim=-1)
#         combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(),
#                                    activation_pred], dim=-1)
#         frame_pred = self.combined_stack(combined_pred)
#         velocity_pred = self.velocity_stack(mel)
#         return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred, instrument_pred
#
#     def run_on_batch(self, batch, parallel_model=None):
#         audio_label = batch['audio']
#         onset_label = batch['onset']
#         offset_label = batch['offset']
#         frame_label = batch['frame']
#         velocity_label = batch['velocity']
#         instrument_label = batch['instrument']
#
#         mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
#         if not parallel_model:
#             onset_pred, offset_pred, _, frame_pred, velocity_pred, inst_pred, inst_pred = self(mel)
#         else:
#             onset_pred, offset_pred, _, frame_pred, velocity_pred, inst_pred = parallel_model(mel)
#
#         predictions = {
#             'onset': onset_pred.reshape(*onset_label.shape),
#             'offset': offset_pred.reshape(*offset_label.shape),
#             'frame': frame_pred.reshape(*frame_label.shape),
#             'velocity': velocity_pred.reshape(*velocity_label.shape),
#             'instrument': inst_pred.reshape(*instrument_label.shape)
#         }
#
#         losses = {
#             'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
#             'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
#             'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
#             'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label),
#             'loss/instrument': F.binary_cross_entropy(predictions['instrument'], instrument_label)
#         }
#
#         return predictions, losses
#
#     def velocity_loss(self, velocity_pred, velocity_label, onset_label):
#         denominator = onset_label.sum()
#         if denominator.item() == 0:
#             return denominator
#         else:
#             return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

