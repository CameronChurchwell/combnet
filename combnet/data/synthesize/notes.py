from random import randint, uniform
import tqdm
import pretty_midi

import combnet

def generate_midi_file(midi_file, n_notes):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    start_time = 0.1
    for _ in range(randint(1, n_notes)):
        pitch = randint(60, 71)  # C4, B4
        duration = uniform(0.2, 1.0)
        velocity = randint(50, 100)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=start_time + duration)
        instrument.notes.append(note)
        start_time += duration

    pm.instruments.append(instrument)
    pm.write(str(midi_file))


def notes(n=1000):
    """
    Generate the notes synthetic dataset using pyfluidsynth and pretty_midi
    """
    dataset_dir = combnet.DATA_DIR / 'notes'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    #TODO add random seeding
    for i in tqdm.tqdm(range(0, n), desc='synthesizing notes dataset', total=n, dynamic_ncols=True):
        stem = f'{i:04d}'
        midi_file = dataset_dir / f'{stem}.midi'
        n_notes = randint(3, 10)
        generate_midi_file(midi_file, n_notes)

        label_file = dataset_dir / f'{stem}-labels.pt'
        combnet.data.synthesize.from_midi_to_labels(midi_file, label_file)

        wav_file = dataset_dir / f'{stem}.wav'
        combnet.data.synthesize.from_midi_to_wav(midi_file, wav_file)