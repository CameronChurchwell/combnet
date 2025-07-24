import torchutil
import combnet
from .chords import chords
from .notes import notes
import fluidsynth
from functools import lru_cache
import torch
import torchaudio
from pathlib import Path
from platformdirs import user_cache_dir


###############################################################################
# Synthesize datasets
###############################################################################


@torchutil.notify('synthesize')
def datasets(datasets=combnet.DATASETS):
    """Synthesize datasets"""
    if 'chords' in datasets:
        chords()
    if 'notes' in datasets:
        notes()

###############################################################################
# Utilities
###############################################################################

@lru_cache
def get_softsynth_file():
    dir_path = Path(user_cache_dir()) / 'combnet'
    sf = dir_path / 'GeneralUser GS 1.44 SoftSynth' / 'GeneralUser GS SoftSynth v1.44.sf2'
    if not sf.exists():
        torchutil.download.zip(
            'https://schriscollins.website/wp-content/uploads/2022/01/GeneralUser_GS_1.44-SoftSynth.zip',
            dir_path,
            use_headers=True,
        )
    assert sf.exists()
    return sf

# Seems to be that state doesn't reset
# Eventually you get errors related to polyphony...
# So for now I won't cache this
# @lru_cache
def get_synth():
    return fluidsynth.Synth(samplerate=combnet.SAMPLE_RATE)

# And apparently there are issues if you try to select a program loaded by a different synth object...
# @lru_cache
def get_softsynth(synth):
    return synth.sfload(str(get_softsynth_file()))

def from_midi_to_wav(
    midi_file,
    wav_file,
    instrument=1,
):
    import pretty_midi
    fl = get_synth()
    soundfont = get_softsynth(synth=fl)
    fl.program_select(0, soundfont, 0, instrument)
    fl.play_midi_file(str(midi_file))

    pm = pretty_midi.PrettyMIDI(str(midi_file))
    n_samples = int(pm.get_end_time()*combnet.SAMPLE_RATE)

    audio = fl.get_samples(n_samples)[::2].astype('float32')
    audio = torch.tensor(audio)
    audio /= abs(audio).max() # normalize to [-1, 1]
    if audio.dim() == 1:
        audio = audio[None]
    torchaudio.save(wav_file, audio, sample_rate=combnet.SAMPLE_RATE)
    # I'll leave these in just in case...
    fl.play_midi_stop()
    fl.router_clear()
    fl.all_sounds_off(0)
    fl.get_samples(combnet.SAMPLE_RATE*10)
    # If we can't cache these, next best thing is to clean up
    fl.sfunload(soundfont)
    fl.delete()

def from_midi_to_labels(
    midi_file,
    label_file
):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_file))
    assert (len(pm.instruments) == 1), len(pm.instruments)
    sr = combnet.SAMPLE_RATE
    assert sr % 5 == 0
    ws = sr // 5
    hs = sr // 10
    chroma = (torch.tensor(pm.get_chroma(sr)) > 0).to(torch.float32)
    chroma = torch.nn.functional.pad(chroma, (ws//2, ws//2))
    labels = torch.nn.functional.max_pool1d(chroma, ws, hs).to(torch.float32)
    torch.save(labels, label_file)

