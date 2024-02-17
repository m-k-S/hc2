from midiutil import MIDIFile
import pygame
from pydub import AudioSegment
import os, re

def parse_input(input_str):
    # Split the input string by newline, then further split by chord symbols to handle cases without newline characters
    parts = input_str.replace('\n', ' ').split(' ')
    arrays = []
    current_numbers = []  # Temporary list to hold numbers for the current chord

    for part in parts:
        if ':' in part:  # Indicates the start of a new chord symbol
            if current_numbers:  # If there are numbers collected for a previous chord, add them to the arrays list
                arrays.append(current_numbers)
                current_numbers = []  # Reset for the next chord

        else:
            try:
                # Attempt to convert the part into an integer and add to the current list of numbers
                number = int(part)
                current_numbers.append(number)
            except ValueError:
                # If conversion fails, it means this part is not a number and can be ignored
                pass
    # After loop, add the last collected numbers to the arrays list, if any
    if current_numbers:
        arrays.append(current_numbers)
    return arrays


def chord_progression_to_midi(progression):
    progression = re.sub(r'[^0-9\n]+', '', progression)
    return progression
    # progression = progression.split("\n")
    # return [[int(j) for j in i.split(": ")[1].split(" ")] for i in progression.split("\n")]

def save_to_midi(progression, save_path="output.mid"):
    track    = 0
    channel  = 0
    time     = 0  # In beats
    duration = 2  # In beats
    tempo    = 40  # In BPM
    volume   = 50  # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created automatically)
    MyMIDI.addTempo(track, time, tempo)

    for i, chord in enumerate(parse_input(progression)):
        for note in chord:
            MyMIDI.addNote(track, channel, int(note), time + (2*i), duration, volume)

    # And write it to disk.
    with open(save_path, "wb") as output_file:
        MyMIDI.writeFile(output_file)

def midi_to_mp3(soundfont_path='', inpath='output.mid', outpath='output.mp3'):
    wav_file = outpath.replace('.mp3', '.wav')

    print (soundfont_path, inpath, outpath, wav_file)
    os.system(f'fluidsynth -ni {soundfont_path} {inpath} -F {wav_file} -r 44100')

    # Convert WAV to MP3 using pydub
    audio = AudioSegment.from_wav(wav_file)
    audio.export(outpath, format='mp3')

    # Remove temporary WAV file
    os.remove(wav_file)

def play_midi(path='output.mid'):
    pygame.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)  # wait 1 second

if __name__=="__main__":
    # progression = "G7: 31 43 59 62 65 74\n Bb13(#11)(9): 34 50 55 56 60 64 72\n A-9: 33 45 59 60 64 67 76\n C/D: 38 50 60 64 67 76\n G(add2): 31 43 59 62 67 69 71 74 79\n G7(b13): 31 43 53 55 63 71 79 83\n E7(#9): 40 52 62 68 74 79 83\n A-7: 33 57 64 67 72 76 79 84\n E7(#9): 28 40 55 56 62 67 71 74 83\n A-7: 33 45 60 64 67 72 76 79 84\n E7(#9): 28 52 62 68 71 74 79 83\n Eb9(#11)(omit3): 39 51 53 61 65 65 69 77 81\n D13sus(b9): 38 50 60 63 67 71 74 79\n E7(#9): 40 52 62 68 71 74 79\n Eb9(#11)(omit3): 39 51 57 61 65 65 69 73 81\n D13sus(b9): 38 50 60 63 67 71 74 79\n C/D: 38 50 60 64 67 72 76\n Bb-11(9): 34 46 56 61 63 68 72 75\n C#-9: 37 52 59 63 68 75\n D#7(#5): 39 51 59 61 63 67 71\n C/D: 38 50 60 64 67 76\n D13sus(b9): 38 50 60 63 67 71 74\n G(add2): 31 43 50 59 62 67 69 71 74\n G7(b13): 31 43 53 55 59 63 71\n E7(#9): 28 40 56 62 67 71\n A-7: 33 45 52 55 60 64 67\n Bb-11(9): 34 46 56 60 61 63 68\n C#-9: 37 52 56 59 63"
    progression = "DbMaj9(#11): 37 53 60 63 67 72 75 C7(#9)(b5): 36 52 58 63 66 70 75 Cb9(#11)(omit3): 35 47 53 57 61 61 65 73 Bb13sus(9): 34 46 56 60 63 67 70 Ab-6: 32 44 51 53 59 68 75 Ab-9: 32 44 51 59 66 70 75 78 82 Bb13sus(9): 34 46 56 60 63 67 70 79 82 EbMaj7: 39 55 62 70 79 CbMaj7: 35 51 54 58 63 70 78 82 Eb6(9): 39 51 58 60 65 67 70 77 82 C7(#9): 36 52 58 63 67 70 79 82 C7(#9)(b5): 36 52 58 66 75 82 A7(#9)(b5): 33 45 55 61 63 72 79 Ab-6: 32 44 51 53 56 59 63 71 80 C7(#9): 36 52 55 58 63 67 70 75 82 Cb9(#11)(omit3): 35 47 57 61 61 69 77 81 Bb13sus(9): 34 46 56 60 63 67 70 79 C7(#9): 36 52 58 63 67 70 79 A-7(b5): 33 45 55 60 63 67 75 79 Ab-6: 32 56 59 63 65 68 71 Db13(#11)(9)(omit3): 37 49 59 63 67 75 82 C7(#9): 36 48 58 64 67 70 75 82 Cb9(#11)(omit3): 35 47 57 61 61 69 77 81 Bb13sus(9): 34 46 56 60 63 70 79 EbMaj7: 39 51 55 62 70 79 82 Bb9sus: 34 46 56 60 63 65 68 72 84 A-7(b5): 33 57 67 72 75 79 84"
    # progression = parse_input(progression)
    save_to_midi(progression, save_path="output.mid")
    midi_to_mp3(soundfont_path='soundfront.sf2', inpath='output.mid', outpath='output.mp3')
    # play_midi('output.mid')

