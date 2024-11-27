import resampy
import soundfile
import torchutil
import torchaudio

import combnet

SEMITONES = [
    'Ab',
    'A',
    'Bb',
    'B',
    'C',
    'Db',
    'D',
    'Eb',
    'E',
    'F',
    'Gb',
    'G'
]

###############################################################################
# Pitch-shifting data augmentation
###############################################################################


def from_audio(audio, sample_rate, shift):
    """Perform pitch-shifting data augmentation on audio"""

    cents = shift * 100

    audio, _ = torchaudio.sox_effects.apply_effects_tensor(
        audio,
        sample_rate,
        [
            ['pitch', f'{cents}'],
            ['rate', str(int(combnet.SAMPLE_RATE))],
            ['dither']
        ]
    )

    return audio

    # ratio = 2 ** (shift / 12)


    # # Augment audio
    # augmented = resampy.resample(audio, int(ratio * sample_rate), sample_rate)

    # return augmented
    # Resample to combnet sample rate
    # return resampy.resample(augmented, sample_rate, combnet.SAMPLE_RATE)

def from_key(key, shift):
    """Perform pitch-shifting data augmentation on key"""
    base_key = key.split(' ')[0]
    assert base_key in SEMITONES, f'base_key is "{base_key}"'
    index = SEMITONES.index(base_key)
    new_index = (index + shift) % len(SEMITONES)
    new_base_key = SEMITONES[new_index]
    new_key = new_base_key + ' ' + key.split(' ')[1]
    return new_key

def from_audio_file(audio_file, shift):
    """Perform pitch-shifting data augmentation on audio file"""
    # audio, sr = soundfile.read(str(audio_file))
    # audio = audio.T
    audio, sr = torchaudio.load(audio_file)
    if len(audio.shape) == 2:
        audio = audio.sum(0, keepdims=True)
    return from_audio(audio, sr, shift)

def from_key_file(key_file, shift):
    """Perform pitch-shifting data augmentation on key file"""
    with open(key_file, 'r') as f:
        key = f.read()
    return from_key(key, shift)

def from_file_to_file(audio_file, key_file, output_file, output_key_file, shift):
    """Perform pitch-shifting data augmentation on audio file and save"""
    augmented_audio = from_audio_file(audio_file, shift)
    augmented_key = from_key_file(key_file, shift)
    torchaudio.save(str(output_file), augmented_audio, combnet.SAMPLE_RATE)
    # soundfile.write(str(output_file), augmented_audio, combnet.SAMPLE_RATE)
    with open(output_key_file, 'w+') as f:
        f.write(augmented_key)


def from_files_to_files(
    audio_files, 
    key_files,
    output_files,
    output_key_files,
    shifts
):
    """Perform pitch-shifting data augmentation on audio files and key files"""
    zipped = list(zip(audio_files, key_files, output_files, output_key_files, shifts))
    torchutil.multiprocess_iterator(
        wrapper,
        zipped,
        'Augmenting pitch',
        total=len(audio_files),
        num_workers=combnet.NUM_WORKERS)


###############################################################################
# Utilities
###############################################################################


def wrapper(item):
    from_file_to_file(*item)