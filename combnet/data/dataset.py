import json
import warnings
from pathlib import Path

# import accelerate
import numpy as np
import torch
import torchaudio

from random import randint

import combnet


###############################################################################
# Dataset
###############################################################################

beadgcf = ["B", "E", "A", "D", "G", "C", "F"]
beadgcf_ext = beadgcf[:]
for note in beadgcf:
    beadgcf_ext.append(note +"b")

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name_or_files,
        partition=None,
        features=['audio'],
        memory_caching=combnet.MEMORY_CACHING):
        self.features = features
        self.metadata = Metadata(
            name_or_files,
            partition=partition)
        self.cache = self.metadata.cache_dir
        self.stems = self.metadata.stems
        self.audio_files = self.metadata.audio_files
        self.lengths = self.metadata.lengths
        self.memory_caching = False
        self.partition = partition
        if memory_caching:
            self.memory_cache = []
            for index in range(0, len(self)):
                self.memory_cache.append(self[index])
        self.memory_caching = memory_caching
        if self.metadata.name == 'timit':
            speakers_file = self.metadata.data_dir / 'speakers.json'
            with open (speakers_file, 'r') as f:
                speakers = json.load(f)
            self.speakers = speakers

    def __getitem__(self, index):
        """Retrieve the indexth item"""

        if self.memory_caching:
            return self.memory_cache[index]

        stem = self.stems[index]

        feature_values = []
        if isinstance(self.features, str):
            self.features = [self.features]
        for feature in self.features:

            # Load audio
            if feature == 'audio':
                audio = combnet.load.audio(self.audio_files[index])
                audio = audio.mean(0, keepdim=True)
                feature_values.append(audio)

            elif feature == 'random-audio-frame':
                #TODO generalize
                frame_size = 3200
                audio = combnet.load.audio(self.audio_files[index])
                audio = audio.mean(0, keepdim=True)
                start = randint(0, audio.shape[-1]-frame_size)
                audio = audio[..., start:start+frame_size]
                feature_values.append(audio)

            elif feature == 'random-audio-frame-noised':
                #TODO generalize
                audio = combnet.load.audio(self.audio_files[index])
                audio = audio.mean(0, keepdim=True)
                if self.partition == 'train':
                    frame_size = 3200
                    start = randint(0, audio.shape[-1]-frame_size)
                    audio = audio[..., start:start+frame_size]
                    random_scalar = (torch.rand(1).squeeze()-0.5)/0.5*0.2
                    audio *= random_scalar
                feature_values.append(audio)

            # elif feature == 'audio-tensor':
            #     audio = torch.load(self.audio_files[index].with_suffix('.pt'))
            #     audio = audio.mean(0, keepdim=True)
            #     feature_values.append(audio)

            elif feature == 'highpass_audio':
                audio = combnet.load.audio(self.cache / (stem + '-highpass_audio.wav'))
                audio = audio.mean(0, keepdim=True)
                feature_values.append(audio)

            # Add stem
            elif feature == 'stem':
                feature_values.append(stem)

            # Add filename
            elif feature == 'audio_file':
                feature_values.append(self.audio_files[index])

            elif feature == 'key_file':
                file = self.metadata.data_dir / (self.stems[index] + '.key')
                feature_values.append(file)

            elif feature == 'key':
                file = self.metadata.data_dir / (self.stems[index] + '.key')
                with open(file, 'r') as f:
                    feature_values.append(f.read())

            elif feature == 'chord_file':
                file = self.metadata.data_dir / (self.stems[index] + '.chord')
                feature_values.append(file)

            elif feature == 'chord':
                file = self.metadata.data_dir / (self.stems[index] + '.chord')
                with open(file, 'r') as f:
                    feature_values.append(f.read())

            elif feature == 'quality_file':
                file = self.metadata.data_dir / (self.stems[index] + '.quality')
                feature_values.append(file)

            elif feature == 'quality':
                file = self.metadata.data_dir / (self.stems[index] + '.quality')
                with open(file, 'r') as f:
                    feature_values.append(f.read())

            elif feature == 'notes_file':
                file = self.metadata.data_dir / (self.stems[index] + '.notes.json')
                feature_values.append(file)

            elif feature == 'notes':
                file = self.metadata.data_dir / (self.stems[index] + '.notes.json')
                with open(file, 'r') as f:
                    feature_values.append(json.load(f))

            elif feature == 'notes_vector':
                file = self.metadata.data_dir / (self.stems[index] + '.notes.json')
                with open(file, 'r') as f:
                    notes = json.load(f)
                vec = torch.zeros(len(beadgcf_ext))
                for note in notes:
                    vec[beadgcf_ext.index(note)] = 1.
                feature_values.append(vec)

            elif feature == 'class':
                if self.metadata.name in ['giantsteps', 'giantsteps_mtg']:
                    file = self.metadata.data_dir / (self.stems[index] + '.key')
                    with open(file, 'r') as f:
                        feature_values.append(torch.tensor(combnet.CLASS_MAP[f.read()]))
                elif self.metadata.name in ['chords']:
                    file = self.metadata.data_dir / (self.stems[index] + '.chord')
                    with open(file, 'r') as f:
                        feature_values.append(torch.tensor(combnet.CLASS_MAP[f.read()]))
                elif self.metadata.name in ['timit']:
                    speaker_id = self.stems[index].split('-')[0]
                    feature_values.append(torch.tensor(combnet.CLASS_MAP[speaker_id]))
                else:
                    raise ValueError(f'class is not a supported feature for dataset {self.metadata.name}')

            elif feature == 'labels':
                if self.metadata.name in ['maestro', 'notes']:
                    file = self.metadata.data_dir / (self.stems[index] + '-labels.pt')
                    feature_values.append(torch.load(file))
                else:
                    raise ValueError(f'labels is not a supported feature for dataset {self.metadata.name}')

            # Add length
            elif feature == 'length':
                try:
                    feature_values.append(feature_values[-1].shape[-1])
                except AttributeError:
                    feature_values.append(len(feature_values[-1]))

            # Add input representation
            else:
                feature_values.append(
                    torch.load(self.cache / f'{stem}-{feature}.pt', weights_only=True))

        return feature_values

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // combnet.BUCKETS

        # Get indices in order of length
        indices = np.argsort(self.lengths)
        lengths = np.sort(self.lengths)

        # Split into buckets based on length
        buckets = [
            np.stack((indices[i:i + size], lengths[i:i + size])).T
            for i in range(0, len(self), size)]

        # Concatenate partial bucket
        if len(buckets) == combnet.BUCKETS + 1:
            residual = buckets.pop()
            buckets[-1] = np.concatenate((buckets[-1], residual), axis=0)

        return buckets


###############################################################################
# Utilities
###############################################################################


class Metadata:

    def __init__(
        self,
        name_or_files,
        partition=None,
        overwrite_cache=False): #TODO change back to False
        """Create a metadata object for the given dataset or sources"""
        lengths = {}

        # Create dataset from string identifier
        if isinstance(name_or_files, str):
            self.name = name_or_files
            self.data_dir = combnet.DATA_DIR / self.name
            self.cache_dir = combnet.CACHE_DIR / self.name

            if not self.cache_dir.exists():
                self.cache_dir.mkdir()

            # Get stems corresponding to partition
            partition_dict = combnet.load.partition(self.name)
            if partition is not None:
                self.stems = partition_dict[partition]
                lengths_file = self.cache_dir / f'{partition}-lengths.json'
            else:
                self.stems = sum(partition_dict.values(), start=[])
                lengths_file = self.cache_dir / f'lengths.json'

            # Get audio filenames
            self.audio_files = [
                self.data_dir / (stem + '.wav') for stem in self.stems]

            # Maybe remove previous cached lengths
            if overwrite_cache:
                lengths_file.unlink(missing_ok=True)

            # Load cached lengths
            if lengths_file.exists():
                with open(lengths_file, 'r') as f:
                    lengths = json.load(f)

        # Create dataset from a list of audio filenames
        else:
            self.name = '<list of files>'
            self.audio_files = name_or_files
            self.stems = [
                Path(file).parent / Path(file).stem
                for file in self.audio_files]
            self.cache_dir = None

        if not lengths:

            # Compute length in frames
            for stem, audio_file in zip(self.stems, self.audio_files):
                info = torchaudio.info(audio_file)
                length = int(
                    info.num_frames * (combnet.SAMPLE_RATE / info.sample_rate)
                ) // combnet.HOPSIZE

                lengths[stem] = length

            # Maybe cache lengths
            if self.cache_dir is not None:
                with open(lengths_file, 'w+') as file:
                    json.dump(lengths, file)

        # Match ordering
        try:
            (
                self.audio_files,
                self.stems,
                self.lengths
                ) = zip(*[
                (file, stem, lengths[stem])
                for file, stem in zip(self.audio_files, self.stems)
                if stem in lengths
            ])
        except ValueError:
            raise ValueError("You probably have to clear the lengths cache file")

    def __len__(self):
        return len(self.stems)