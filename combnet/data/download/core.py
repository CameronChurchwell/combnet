import json
import shutil
import torchutil
import requests
import combnet
from tqdm import tqdm
import time
from io import BytesIO, StringIO
import torchaudio
from pathlib import Path
import tarfile
import csv

from . import timit

###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
def datasets(datasets=combnet.DATASETS):
    """Download datasets"""
    if 'giantsteps' in datasets:
        giantsteps()
    if 'giantsteps_mtg' in datasets:
        giantsteps_mtg()
    if 'maestro' in datasets:
        maestro()
    if 'timit' in datasets:
        timit.download()
        timit.format()


def giantsteps():
    dataset_dir = combnet.DATA_DIR / 'giantsteps'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/GiantSteps/giantsteps-key-dataset/archive/refs/heads/master.tar.gz"

    audio_files = []

    def key_file_filter(file: tarfile.TarInfo, _):
        path_ignoring_repo = Path(*Path(file.name).parts[1:])
        if str(path_ignoring_repo).startswith('annotations/key/'):
            audio_files.append(path_ignoring_repo.stem + '.mp3')
            file.name = str(path_ignoring_repo.name) # strip parent dirs from download
            return file
        return None

    with requests.get(repo_url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|gz') as tstream:
            tstream.extractall(dataset_dir, filter=key_file_filter)

    audio_url = 'http://www.cp.jku.at/datasets/giantsteps/backup/'

    # download audio mp3 files
    for file in tqdm(audio_files, 'downloading audio data for giantsteps', len(audio_files), dynamic_ncols=True):
        wav_file = dataset_dir / (Path(file).stem + '.wav')
        file_download = requests.get(audio_url + file)
        if file_download.status_code != 200:
            import pdb; pdb.set_trace()
            raise ValueError(f'download error {file_download.status_code}')
        with BytesIO(file_download.content) as mp3_data:
            audio, sr = torchaudio.load(mp3_data)
        torchaudio.save(wav_file, audio, sr)
        time.sleep(0.5) # Courtesy delay


def giantsteps_mtg():
    dataset_dir = combnet.DATA_DIR / 'giantsteps_mtg'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    url = "https://api.github.com/repos/GiantSteps/giantsteps-mtg-key-dataset/contents/annotations/annotations.txt"
    annotations_metadata = requests.get(url)
    annotations_metadata.raise_for_status()

    url = annotations_metadata.json()['download_url']
    annotations_download = requests.get(url)
    annotations_download.raise_for_status()

    annotations = StringIO(annotations_download.content.decode('utf-8'))

    annotations = csv.reader(annotations, delimiter='\t', )
    next(annotations) # skip header

    audio_files = []
    for row in annotations:
        name = row[1].rstrip()
        if int(row[2]) < 2:
            continue
        if name not in combnet.GIANTSTEPS_KEYS:
            if name in combnet.KEY_MAP:
                name = combnet.KEY_MAP[name]
            else:
                if '-' in name or '/' in name or name in [' ', '']:
                    continue # junk files
                raise ValueError('unknown file contents')
        stem = row[0] + ".LOFI"
        audio_files.append(stem + ".mp3")
        with open(dataset_dir / (stem + '.key'), 'w+') as f:
            f.write(name)

    audio_url = 'http://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/'

    # download audio mp3 files
    for file in tqdm(audio_files, 'downloading audio data for giantsteps_mtg', len(audio_files), dynamic_ncols=True):
        wav_file = dataset_dir / (Path(file).stem + '.wav')
        file_download = requests.get(audio_url + file)
        if file_download.status_code != 200:
            import pdb; pdb.set_trace()
            raise ValueError(f'download error {file_download.status_code}')
        with BytesIO(file_download.content) as mp3_data:
            audio, sr = torchaudio.load(mp3_data)
        torchaudio.save(wav_file, audio, sr)
        time.sleep(0.5) # Courtesy delay


def maestro(max_files = None):
    SAMPLE_RATE = 22_050
    data_dir = combnet.DATA_DIR / 'maestro'
    data_dir.mkdir(exist_ok=True, parents=True)

    # print('Downloading Maestro MIDI data...')
    # url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
    # torchutil.download.zip(url, data_dir)
    # print('Download complete, synthesizing select audio')

    source_dir = data_dir / 'maestro-v3.0.0'

    with open(source_dir / 'maestro-v3.0.0.json', 'r') as f:
        metadata = json.load(f)

    titles = metadata['canonical_title']
    titles = {int(k): v for k, v in titles.items()}
    titles = [titles[i] for i in range(0, len(titles))] # convert to list

    selected_indices = []
    selected_titles = set()
    for i, title in enumerate(titles):
        title = title.lower()
        if title not in selected_titles:
            selected_indices.append(i)
            selected_titles.add(title)

    midi_files = metadata['midi_filename']
    midi_files = {int(k): v for k, v in midi_files.items()}
    midi_files = [midi_files[i] for i in selected_indices[:max_files]] # convert to list

    splits = metadata['split']
    splits = {int(k): v for k, v in splits.items()}
    splits = [splits[i] for i in selected_indices[:max_files]] # convert to list

    select_midi_files = []
    partition = {'train': [], 'valid': [], 'test': []}
    for i, (midi_file, split) in enumerate(zip(midi_files, splits)):
        if split == 'train':
            partition['train'].append(f'{i:04d}')
        if split == 'validation':
            partition['valid'].append(f'{i:04d}')
        if split == 'test':
            partition['test'].append(f'{i:04d}')
        new_filename = data_dir/f'{i:04d}.midi'
        shutil.copy(source_dir/midi_file, new_filename)
        select_midi_files.append(new_filename)

    partition_file = combnet.PARTITION_DIR / 'maestro.json'
    with open(partition_file, 'w+') as f:
        json.dump(partition, f)

    for midi_file in torchutil.iterator(select_midi_files, 'Synthesizing midi', total=len(select_midi_files)):
        audio_file = data_dir / f'{midi_file.stem}.wav'
        label_file = data_dir / f'{midi_file.stem}-labels.pt'
        combnet.data.synthesize.from_midi_to_wav(midi_file, audio_file, instrument=1)
        combnet.data.synthesize.from_midi_to_labels(midi_file, label_file)