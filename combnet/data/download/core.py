import torchutil
import requests
import combnet
from tqdm import tqdm
import time
from io import BytesIO
import torchaudio
from pathlib import Path


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
def datasets(datasets=combnet.DATASETS):
    """Download datasets"""
    if 'giantsteps' in datasets:
        giantsteps()


def giantsteps():
    base_url = 'https://api.github.com/repos/GiantSteps/giantsteps-key-dataset/contents/'

    dataset_dir = combnet.DATA_DIR / 'giantsteps'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    key_dir_url = base_url + 'annotations/key/'
    # Get dir contents
    response = requests.get(key_dir_url)

    audio_files = []

    # Download key files
    for file in tqdm(response.json(), 'downloading key data', len(response.json()), dynamic_ncols=True):
        audio_files.append(file['name'][:-3] + 'mp3')
        with open(dataset_dir / file['name'], 'wb+') as f:
            file_download = requests.get(file['download_url'])
            if file_download.status_code != 200:
                raise ValueError(f'download error {file_download.status_code}')
            f.write(file_download.content)

    audio_url = 'http://www.cp.jku.at/datasets/giantsteps/backup/'

    # audio_dir = dataset_dir / 'audio'
    # audio_dir.mkdir(parents=True, exist_ok=True)


    # download audio mp3 files
    for file in tqdm(audio_files, 'downloading audio data', len(audio_files), dynamic_ncols=True):
        wav_file = dataset_dir / (Path(file).stem + '.wav')
        file_download = requests.get(audio_url + file)
        if file_download.status_code != 200:
            import pdb; pdb.set_trace()
            raise ValueError(f'download error {file_download.status_code}')
        with BytesIO(file_download.content) as mp3_data:
            audio, sr = torchaudio.load(mp3_data)
        torchaudio.save(wav_file, audio, sr)

        time.sleep(0.5) #TODO investigate if this is necessary