import torchutil
import combnet
from .chords import chords
import fluidsynth
import pretty_midi
from functools import lru_cache
import torch
import torchaudio
import numpy as np


###############################################################################
# Synthesize datasets
###############################################################################


@torchutil.notify('synthesize')
def datasets(datasets=combnet.DATASETS):
    """Synthesize datasets"""
    if 'chords' in datasets:
        chords()

###############################################################################
# Utilities
###############################################################################

@lru_cache
def get_softsynth_file():
    sf = combnet.DATA_DIR / 'GeneralUser GS 1.44 SoftSynth' / 'GeneralUser GS SoftSynth v1.44.sf2'
    if not sf.exists():
        torchutil.download.zip(
            'https://schriscollins.website/wp-content/uploads/2022/01/GeneralUser_GS_1.44-SoftSynth.zip',
            combnet.DATA_DIR,
            use_headers=True,
        )
    assert sf.exists()
    return sf

@lru_cache
def get_synth():
    return fluidsynth.Synth(samplerate=combnet.SAMPLE_RATE)

@lru_cache
def get_softsynth():
    return get_synth().sfload(str(get_softsynth_file()))

def get_instrument(instrument=1):
    fl = get_synth()
    soundfont = get_softsynth()
    fl.program_select(0, soundfont, 0, instrument)
    return fl

def from_midi_to_wav(
    midi_file,
    wav_file,
    instrument=1,
    seconds=30
):
    fl = get_instrument(instrument)
    fl.play_midi_file(str(midi_file))

    audio = fl.get_samples(combnet.SAMPLE_RATE*seconds)[::2].astype('float32')
    audio = torch.tensor(audio)
    audio /= abs(audio).max() # normalize to [-1, 1]
    if audio.dim() == 1:
        audio = audio[None]
    torchaudio.save(wav_file, audio, sample_rate=combnet.SAMPLE_RATE)
    fl.play_midi_stop()
    fl.router_clear()
    fl.all_sounds_off(0)
    fl.get_samples(combnet.SAMPLE_RATE*10)

def from_midi_to_labels(
    midi_file,
    label_file,
    seconds=30,
    hopsize=combnet.HOPSIZE
):
    enharmonic_map = {
        'A#': 'Bb',
        'B#': 'C',
        'C#': 'Db',
        'D#': 'Eb',
        'E#': 'F',
        'F#': 'Gb',
        'G#': 'Ab',
    }
    pm = pretty_midi.PrettyMIDI(str(midi_file))
    n_frames = ((seconds * combnet.SAMPLE_RATE) - combnet.WINDOW_SIZE) // combnet.HOPSIZE + 1
    offset_to_center = combnet.WINDOW_SIZE // 2 // combnet.SAMPLE_RATE
    # time_steps =  
    breakpoint()
    # time_steps = np.linspace(, seconds, n_frames)
    ALL_NOTES = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "B", "E", "A", "D", "G"]
    frames = []
    for t in time_steps:
        frame_notes = np.zeros(len(ALL_NOTES))
        assert (len(pm.instruments) == 1), len(pm.instruments)
        for instrument in pm.instruments:
            for note in instrument.notes:
                if note.start <= t < note.end: # this is a bit naïve
                    name_with_octave = pretty_midi.note_number_to_name(note.pitch)
                    name = name_with_octave.rstrip('0123456789').upper()
                    name = enharmonic_map[name] if name in enharmonic_map else name
                    index = ALL_NOTES.index(name)
                    frame_notes[index] = 1
        frames.append(frame_notes[:, None])
    labels = np.concatenate(frames, axis=1)
    labels = torch.tensor(labels)
    breakpoint()
    torch.save(labels, label_file)

