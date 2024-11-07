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
        time.sleep(0.5) #TODO investigate if this is necessary


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
        time.sleep(0.5) #TODO investigate if this is necessary