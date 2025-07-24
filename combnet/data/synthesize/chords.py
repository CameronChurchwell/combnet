from random import randint, random, choice, shuffle
import torch
import torchaudio
import numpy as np
import copy
import tqdm
import json

import combnet

allowed_instruments = [
    0, #Stereo Grand
    1, #Bright Grand
    2, #Electric Grand
    # 4, #Tine Electric Piano
    # 5, #FM Electric Piano
    # 6, #Harpsichord
    # 9, #Glockenspiel
    11, #Vibraphone
    12, #Marimba
    # 13, #Xylophone
    19, #Pipe Organ
    # 21, #Accordian
    # 22, #Harmonica
    # 24, #Nylon Guitar
    # 25, #Steel Guitar
    # 32, #Acoustic Bass
    # 35, #Fretless Bass
    # 36, #Slap Bass 1
    # 38, #Synth Bass 1
    40, #Violin
    # 41, #Viola
    # 42, #Cello
    # 50, #Synth Strings 1
    # 51, #Synth Strings 2
    # 53, #Voice Oohs
    56, #Trumpet
    57, #Trombone
    # 58, #Tuba
    60, #French Horns
    65, #Alto Sax
    # 66, #Tenor Sax
    # 67, #Baritone Sax
    # 68, #Oboe
    # 70, #Bassoon
    71, #Clarinet
    73, #Flute
]

beadgcf = ["B", "E", "A", "D", "G", "C", "F"]
beadgcf_ext = beadgcf[:]
for note in beadgcf:
    beadgcf_ext.append(note +"b")
naturalNoteNums = {"C":36, "D":38, "E":40, "F":41, "G":43, "A":45, "B":47}

enharmonic_map = {
    'A#': 'Bb',
    'B#': 'C',
    'C#': 'Db',
    'D#': 'Eb',
    'E#': 'F',
    'F#': 'Gb',
    'G#': 'Ab',
}

def getNoteNum(name):
    natural = name[:1].upper()
    extra = name[1:]
    assert len(extra) <= 1
    base = naturalNoteNums[natural]
    if extra == 'b':
        base -= 1
    elif extra == '#':
        base += 1
    elif extra == '':
        pass
    else:
        raise ValueError(f'Unknown note {name}')
    return base

def fixInversion(nums):
    for i in range(1, len(nums)):
        if nums[i] < nums[i-1]:
            nums[i] += 12
    return nums

def getNoteNums(names):
    octave_shift = choice([0, 12, 24, 36])
    # octave_shift = choice([12, 24, 36])
    # octave_shift = 0
    # octave_shift = 36
    nums = [getNoteNum(n) + octave_shift for n in names]
    return fixInversion(nums)

class Key():
    name = ""
    notes = []

    def __init__(self, name=None, notes=None):
        if name is None:
            self.name = "C"
        else:
            self.name = name
        if notes is None:
            self.notes = ["C", "D", "E", "F", "G", "A", "B"]
        else:
            self.notes = notes

    @classmethod
    def fromName(cls, name):
        self = cls()
        self.name = name

        keyOfC = Key()

        offsetFromC = beadgcf_ext.index(name) - beadgcf_ext.index("C")

        if offsetFromC == 0: #C
            self.notes = keyOfC.notes
        elif offsetFromC > 0: #Flat
            flattedNotes = beadgcf[0:offsetFromC]
            self.notes = keyOfC.notes
            for note in flattedNotes:
                pos = self.notes.index(note)
                self.notes[pos] += "b"
        elif offsetFromC < 0: #Sharp
            sharpedNotes = beadgcf[offsetFromC:]
            self.notes = keyOfC.notes
            for note in sharpedNotes:
                pos = self.notes.index(note)
                self.notes[pos] += "#"

        startNotePos = self.notes.index(name)
        self.notes = self.notes[startNotePos:] + self.notes[:startNotePos]
        return self

    def getNotes(self):
        return self.notes

    def getMajorChord(self):
        return [self.notes[0], self.notes[2], self.notes[4]]

    def getMinorChord(self):
        newKey = self.getDorianKey()
        return newKey.getMajorChord()

    def get7Chord(self):
        return [self.notes[0], self.notes[2], self.notes[4], self.notes[6]]

    def getDorianKey(self):
        newKey = self.getModeKey(2)

        newKey.name = self.name + "m7"
        try:
            newKey = newKey.startingOn(self.name)
        except:
            import pdb; pdb.set_trace()
        return newKey

    def getMixolydianKey(self):
        newKey = self.getModeKey(1)

        newKey.name = self.name + "7"
        newKey = newKey.startingOn(self.name)
        return newKey

    def getModeKey(self, modeNum):
        currentKeyIndex = beadgcf_ext.index(self.name)
        modeKeyIndex = (currentKeyIndex + modeNum) % len(beadgcf_ext)
        try:
            modeKeyName = beadgcf_ext[modeKeyIndex]
        except:
            import pdb; pdb.set_trace()

        newKey = Key().fromName(modeKeyName)

        return newKey

    def startingOn(self, startNoteName):
        startNotePos = self.notes.index(startNoteName)

        newKey = copy.deepcopy(self)
        newKey.notes = newKey.notes[startNotePos:] + newKey.notes[:startNotePos]

        return newKey

# Render a note
def get_note(note, vel, bend, start, hold, end, fl):
    from numpy import concatenate
    from scipy.signal import resample_poly

    # For some reason pyfluidsynth doesn't respect specified sample/rate and channels
    # The ::2 is to get only one channel, by default it produces sound at 44100 
    # I also scale by 1000 since the samples are too loud
    def get_samples(t):
        return fl.get_samples(int(t*combnet.SAMPLE_RATE))[::2].astype('float32')/1000

    # Set the bend
    fl.pitch_bend(0, bend)

    # Get front silence samples
    s0 = get_samples(start) if start > 0 else []

    # Start a note and get its samples
    fl.noteon(0, note, vel)
    s1 = get_samples(hold)

    # Turn off the note and get end silence
    fl.noteoff(0, note)
    s2 = get_samples(end) if end > 0 else []

    # Undo the bend
    fl.pitch_bend( 0, 0)

    # Resample to desired rate
    return concatenate((s0,s1,s2))


# Make a bunch of notes with random pitch bend and velocity
def generate_notes(chord, fl):
    z = []
    fl.all_sounds_off(0) # Important, empty any lingering notes from the buffer
    notes = getNoteNums(chord)
    # print(notes, chord_name, is_major)
    shuffle(notes)
    for i in range(0, len(notes)):
        note = notes[i]

        # Define start silence/duration/end silence times in seconds
        start = random()*0.1
        hold = (random() ** 0.2) * 2 + 0.2
        end = random()*.1

        # Define pitch bend and key velocity
        bend = randint( -10, 10) # make it subtle, can go from -8192 to 8192
        velocity = randint( 80, 127) # 0 to 127

        # Make the note and add it to the list
        z += [get_note(note, velocity, bend, start, hold, end, fl=fl)]

    max_len = max(len(ch) for ch in z)
    out = np.zeros(max_len, dtype=z[0].dtype)
    h = np.hanning(32)[16:] # prevent severe discontinuities
    for ch in z:
        ch[-16:] *= h
        out[:len(ch)] += ch
    return out

ALL_CHORDS = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "B", "E", "A", "D", "G"]
def chords(n=1000, chords=ALL_CHORDS, task='categorical'):
    """
    Generate the chords synthetic dataset using pyfluidsynth
    """
    dataset_dir = combnet.DATA_DIR / 'chords'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm.tqdm(range(0, n), desc='synthesizing chords dataset', total=n, dynamic_ncols=True):
        stem = str(i)
        chord = choice(chords)
        is_major = True
        key = Key.fromName(chord)
        notes = key.getMajorChord() if is_major else key.getMinorChord()
        instrument = choice(allowed_instruments)
        fl = combnet.data.synthesize.get_synth()
        soundfont = combnet.data.synthesize.get_softsynth(synth=fl)
        fl.program_select(0, soundfont, 0, instrument)
        chord_audio = torch.tensor(generate_notes(notes, fl=fl), dtype=torch.float32)[None]
        chord_audio /= abs(chord_audio).max() # normalize to [-1, 1]
        quality = 'major' if is_major else 'minor'
        label = chord + ('m' if not is_major else '')
        torchaudio.save(dataset_dir / f'{stem}.wav', chord_audio, combnet.SAMPLE_RATE)
        with open(dataset_dir / f'{stem}.notes.json', 'w+') as f:
            notes = [enharmonic_map[note] if note in enharmonic_map else note for note in notes]
            json.dump(notes, f)
        with open(dataset_dir / f'{stem}.quality', 'w+') as f:
            f.write(quality)
        with open(dataset_dir / f'{stem}.chord', 'w+') as f:
            f.write(label)
        with open(dataset_dir / f'{stem}.instrument', 'w+') as f:
            f.write(str(instrument))