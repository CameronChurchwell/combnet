import struct
import tarfile
import json

import pypar

import torchutil
import torchaudio

import combnet

SAMPLE_RATE = 16_000


###############################################################################
# Setup TIMIT
###############################################################################


def download():
    """Prompts user to install TIMIT dataset, and formats dataset if present"""
    source_directory = combnet.DATA_DIR / 'timit'
    source_directory.mkdir(parents=True, exist_ok=True)

    # Get tarball
    possible_files = [
        'timit',
        'timit.tar',
        'timit_LDC93S1.tgz',
        'timit_LDC9321.tar.gz',
        'timit.tgz',
        'timit.tar.gz']
    possible_paths = [source_directory / file for file in possible_files]
    source_exists = [path.exists() for path in possible_paths]
    try:
        chosen_source_idx = source_exists.index(True)
    except ValueError:
        raise FileNotFoundError(
            f'TIMIT dataset not found. Please download TIMIT '
            f'via https://catalog.ldc.upenn.edu/LDC93s1 '
            f'and place it in {source_directory}. '
            f'This program expects one of {possible_paths} to be present.')
    chosen_source = possible_paths[chosen_source_idx]

    # Untar
    with tarfile.open(chosen_source) as tf:
        tf.extractall(combnet.DATA_DIR)


def format():
    """Format TIMIT"""
    data_directory = combnet.DATA_DIR / 'timit'

    # Get files
    sphere_files = list((data_directory / 'TIMIT' / 'TRAIN').rglob('*.WAV'))

    # Convert NIST sphere files to WAV format and create speaker label files
    all_speakers = set()
    for sphere_file in torchutil.iterator(
        sphere_files,
        'Converting TIMIT audio',
        total=len(sphere_files)
    ):
        speaker = sphere_file.parent.name.lower()
        utterance = sphere_file.stem.lower()
        if utterance in ['sa1', 'sa2']:
            continue
        all_speakers.add(speaker)
        out_stem = speaker + '-' + utterance
        audio_file = data_directory / (out_stem + '.wav')
        with open(audio_file, 'wb') as file:
            file.write(sphere_to_wav(sphere_file))
        label_file = sphere_file.with_suffix('.WRD')
        with open(label_file, 'r') as file:
            lines = file.readlines()
        start_sample = int(lines[0].split()[0])
        end_sample = int(lines[-1].split()[1])
        audio, sr = torchaudio.load(audio_file)
        audio = audio[..., start_sample:end_sample]
        audio = audio/abs(audio).max()
        torchaudio.save(audio_file, audio, SAMPLE_RATE)
    with open(data_directory / 'speakers.json', 'w+') as f:
        json.dump(list(sorted(list(all_speakers))), f)


###############################################################################
# Convert sphere files to wav
###############################################################################


def sphere_to_wav(sphere_file):
    """Load sphere file and convert to wav"""
    with open(sphere_file, 'rb') as f:
        header_size = sph_get_header_size(f)
        header = sph_get_header(f, header_size)
        new_header = wav_make_header(header)
        samples = sph_get_samples(f, header_size)
        return new_header + samples


###############################################################################
# Convert IPA phonemes to CMU
###############################################################################


def ipa_to_cmu(rows, backfill=True):
    """Phoneme conversion from IPA to CMU"""
    # Convert
    phones = []
    transposed = list(zip(*rows))
    for phone in transposed[2]:
        try:
            phones.append(combnet.TIMIT_TO_ARCTIC_MAPPING[phone.lower()])
        except KeyError:
            phones.append(pypar.SILENCE)

    # Handle non-one-to-one or context-dependent phoneme conversions
    if backfill:
        backfill_indices = [
            idx for idx, phone in enumerate(phones)
            if phone[:3] == 'bck']
        for i, idx in enumerate(backfill_indices):
            assert (
                phones[idx][3] == '<' and phones[idx][-1] == '>')
            possible_replacements = phones[idx][4:-1].split(',')
            if (
                idx < len(phones) - 1 and
                phones[idx + 1] in possible_replacements
            ):
                phones[idx] = 'bck'
            else:
                phones[idx] = possible_replacements[0]
        for i in range(0, len(phones)):
            if phones[i] == 'bck':
                phones[i] = phones[i + 1]

    # Convert times to seconds
    phone_ends = [int(sample) / 16000 for sample in list(transposed[1])]

    return list(zip(phone_ends, phones))


###############################################################################
# Utilities
###############################################################################


def sph_get_header(sphere_file_object, header_size):
    """Get metadata"""
    if not hasattr(sph_get_header, 'mapping'):
        sph_get_header.mapping = {'i': int, 'r': float, 's': str}
    sphere_file_object.seek(16)
    header = sphere_file_object.read(
        header_size - 16
    ).decode('utf-8').split('\n')
    header = header[:header.index('end_head')]
    header = [
        header_item.split(' ') for header_item in header
        if header_item[0] != ';']
    return {h[0]: sph_get_header.mapping[h[1][1]](h[2]) for h in header}


def sph_get_header_size(sphere_file_object):
    """Get size of metadata in bytes"""
    sphere_file_object.seek(0)
    assert sphere_file_object.readline() == b'NIST_1A\n'
    header_size = int(sphere_file_object.readline().decode('utf-8')[:-1])
    sphere_file_object.seek(0)
    return header_size


def sph_get_samples(sphere_file_object, sphere_header_size):
    """Extract audio from sphere file"""
    sphere_file_object.seek(sphere_header_size)
    return sphere_file_object.read()


def wav_make_header(sph_header):
    """Create wav file header"""
    # Size of audio in bytes
    samples_bytes = sph_header['sample_count'] * sph_header['sample_n_bytes']

    # Create header
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        samples_bytes + 36, # total size
        b'WAVE',
        b'fmt ',
        16,  # fmt size
        1,  # header
        sph_header['channel_count'],  # channels
        sph_header['sample_rate'],  # sample rate
        sph_header['sample_rate'] * sph_header['sample_n_bytes'],  # bps
        sph_header['sample_n_bytes'],  # bytes per sample
        sph_header['sample_n_bytes']*8, # bit depth
        b'data',
        samples_bytes  # size
    )