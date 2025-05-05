import torch
import yapecs

import combnet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import penn
else:
    from combnet import penn

###############################################################################
# Comb Crepe
###############################################################################

class CombCrepe(torch.nn.Sequential):

    def __init__(self, n_filters=128, fused_comb_fn='FusedComb1d', comb_kwargs={}):
        super().__init__()
        fused_comb_fn: combnet.modules.FusedComb1d = getattr(combnet.modules, fused_comb_fn)
        # in_channels = [1, 1024, 128, 128, 128, 256]
        in_channels = [1, n_filters, 128, 128, 128, 256]
        # out_channels = [1024, 128, 128, 128, 256, 512]
        out_channels = [n_filters, 128, 128, 128, 256, 512]

        kernels = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        padding = [(254, 254)] + 5 * [(31, 32)]
        lengths = [256, 128, 64, 32, 16, 8]
        comb_filter = fused_comb_fn(
            in_channels[0],
            out_channels[0],
            sr=combnet.SAMPLE_RATE,
            window_size=kernels[0],
            stride=strides[0],
            min_freq=penn.FMIN,
            max_freq=penn.FMAX,
            min_bin=1,
            max_bin=penn.PITCH_BINS,
            **comb_kwargs
        )
        super().__init__(*(
            ([penn.model.Normalize()] if penn.NORMALIZE_INPUT else []) +
            [
                comb_filter
            ] + 
            [
                CrepeBlock(i, o, k, s, p, l) for i, o, k, s, p, l in
                list(zip(
                    in_channels,
                    out_channels,
                    kernels,
                    strides,
                    padding,
                    lengths
                ))[1:]
            ] +
            [
                penn.model.Flatten(),
                torch.nn.Linear(
                    in_features=2048,
                    out_features=penn.PITCH_BINS)
            ]
        ))
        object.__setattr__(self, 'comb_filter', comb_filter)

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.comb_filter.f]
        exclude = [id(p) for p in groups['f0']]
        groups['main'] = [p for p in self.parameters() if id(p) not in exclude] #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, frames):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        return super().forward(frames)[:, :, None]


###############################################################################
# Utilities
###############################################################################


class CrepeBlock(torch.nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        length):
        layers = (
            torch.nn.ConstantPad1d(padding, 0),
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride),
            torch.nn.ReLU())

        # Maybe add normalization
        if penn.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif penn.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif penn.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)
        else:
            raise ValueError(
                f'Normalization method {penn.NORMALIZATION} is not defined')

        # Add max pooling
        layers += (torch.nn.MaxPool1d(2, 2),)

        # Maybe add dropout
        if penn.DROPOUT is not None:
            layers += (torch.nn.Dropout(penn.DROPOUT),)

        super().__init__(*layers)


# class CombFcnf0(torch.nn.Sequential):

#     def __init__(self, n_filters=128, fused_comb_fn='FusedComb1d', comb_kwargs={}):
#         fused_comb_fn: combnet.modules.FusedComb1d = getattr(combnet.modules, fused_comb_fn)
#         layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
#         comb_filter = fused_comb_fn(
#             1,
#             n_filters,
#             sr=combnet.SAMPLE_RATE,
#             window_size=32,
#             stride=2,
#             min_freq=penn.FMIN,
#             max_freq=penn.FMAX,
#             min_bin=1,
#             max_bin=penn.PITCH_BINS,
#             **comb_kwargs
#         )
#         layers += (
#             # Block(1, 256, 481, (2, 2)),
#             comb_filter,
#             Fcnf0Block(n_filters, 32, 225, (2, 2)),
#             Fcnf0Block(32, 32, 97, (2, 2)),
#             Fcnf0Block(32, 128, 66),
#             Fcnf0Block(128, 256, 35),
#             Fcnf0Block(256, 512, 4),
#             torch.nn.Conv1d(512, penn.PITCH_BINS, 4))
#         super().__init__(*layers)
#         object.__setattr__(self, 'comb_filter', comb_filter)

#     def parameter_groups(self):
#         groups = {}
#         groups['f0'] = [self.comb_filter.f]
#         exclude = [id(p) for p in groups['f0']]
#         groups['main'] = [p for p in self.parameters() if id(p) not in exclude] #+ [self.filters[0].a] + [self.filters[0].g]
#         return groups

#     def forward(self, frames):
#         # shape=(batch, 1, penn.WINDOW_SIZE) =>
#         # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
#         return super().forward(frames[:, :, 16:-15])


# class Fcnf0Block(torch.nn.Sequential):

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         length=1,
#         pooling=None,
#         kernel_size=32):
#         layers = (
#             torch.nn.Conv1d(in_channels, out_channels, kernel_size),
#             torch.nn.ReLU())

#         # Maybe add pooling
#         if pooling is not None:
#             layers += (torch.nn.MaxPool1d(*pooling),)

#         # Maybe add normalization
#         if penn.NORMALIZATION == 'batch':
#             layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
#         elif penn.NORMALIZATION == 'instance':
#             layers += (torch.nn.InstanceNorm1d(out_channels),)
#         elif penn.NORMALIZATION == 'layer':
#             layers += (torch.nn.LayerNorm((out_channels, length)),)
#         else:
#             raise ValueError(
#                 f'Normalization method {penn.NORMALIZATION} is not defined')

#         # Maybe add dropout
#         if penn.DROPOUT is not None:
#             layers += (torch.nn.Dropout(penn.DROPOUT),)

#         super().__init__(*layers)


class CombFcnf0(torch.nn.Sequential):

    def __init__(self, n_filters=256, fused_comb_fn='FusedComb1d', comb_kwargs={}):
        fused_comb_fn: combnet.modules.FusedComb1d = getattr(combnet.modules, fused_comb_fn)
        layers = (penn.model.Normalize(),) if penn.NORMALIZE_INPUT else ()
        comb_kwargs = {'window_size': 32, **comb_kwargs}
        comb_filter = fused_comb_fn(
            1,
            n_filters,
            sr=combnet.SAMPLE_RATE,
            stride=1,
            min_freq=penn.FMIN,
            max_freq=penn.FMAX,
            **comb_kwargs
        )
        layers += (
            # Block(1, 256, 481, (2, 2)),
            Fcnf0CombBlock(1, n_filters, comb_filter, 481, (2, 2)),
            Fcnf0Block(n_filters, 32, 225, (2, 2)),
            Fcnf0Block(32, 32, 97, (2, 2)),
            Fcnf0Block(32, 128, 66),
            Fcnf0Block(128, 256, 35),
            Fcnf0Block(256, 512, 4),
            torch.nn.Conv1d(512, penn.PITCH_BINS, 4))
        super().__init__(*layers)
        object.__setattr__(self, 'comb_filter', comb_filter)

    def parameter_groups(self):
        groups = {}
        groups['f0'] = [self.comb_filter.f]
        exclude = [id(p) for p in groups['f0']]
        groups['main'] = [p for p in self.parameters() if id(p) not in exclude] #+ [self.filters[0].a] + [self.filters[0].g]
        return groups

    def forward(self, frames):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        return super().forward(frames[:, :, 16:-15])


class Fcnf0CombBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        comb_layer,
        length=1,
        pooling=None,
        kernel_size=32):
        layers = (
            comb_layer,
            torch.nn.Sigmoid())

        # Maybe add pooling
        if pooling is not None:
            layers += (torch.nn.MaxPool1d(*pooling),)

        # Maybe add normalization
        if penn.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif penn.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif penn.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)
        else:
            raise ValueError(
                f'Normalization method {penn.NORMALIZATION} is not defined')

        # Maybe add dropout
        if penn.DROPOUT is not None:
            layers += (torch.nn.Dropout(penn.DROPOUT),)

        super().__init__(*layers)

class Fcnf0Block(torch.nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        pooling=None,
        kernel_size=32):
        layers = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size),
            torch.nn.ReLU())

        # Maybe add pooling
        if pooling is not None:
            layers += (torch.nn.MaxPool1d(*pooling),)

        # Maybe add normalization
        if penn.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif penn.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif penn.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)
        else:
            raise ValueError(
                f'Normalization method {penn.NORMALIZATION} is not defined')

        # Maybe add dropout
        if penn.DROPOUT is not None:
            layers += (torch.nn.Dropout(penn.DROPOUT),)

        super().__init__(*layers)