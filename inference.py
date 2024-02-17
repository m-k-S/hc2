import torch
from model import GPTLanguageModel
from data import encode, decode, load_data
from midi import save_to_midi, midi_to_mp3
from tqdm import tqdm
import json
import re, os

def build_inputs(collection):
    key = collection['key']
    chords = "\n ".join(collection['chordSymbols'])
    output = f"[KEY: {key}\n COLLECTION: {chords}\n "
    return output

def format_chords(output):
    return output.rstrip().split("\n")[-1].split("CHORDS:\\n ")[1].replace("\\n", "")

def load_vocab(data_path):
    train_data, val_data, chars, vocab_size, collection = load_data(data_path)
    return train_data, val_data, chars, vocab_size, collection

def load_model(
        weights_path,
        vocab_size,
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.2
        ):

    model = GPTLanguageModel(vocab_size, block_size, n_head, n_layer, n_embd, n_head, dropout)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.eval()
    return m

if __name__ == "__main__":
    if not os.path.exists("generated"):
        os.mkdir("generated")
        os.mkdir("generated/midi")
        os.mkdir("generated/mp3")

    data_file_path = 'data.txt'
    if not os.path.exists(data_file_path):
        import urllib.request
        print("Downloading data.txt...")
        data_url = 'https://www.dropbox.com/scl/fi/2yod9hatuawm8vjd8aox7/data.txt?rlkey=v530uyk0nrpuz8b66nzc41tmo&dl=1'  # Replace this URL with the actual URL
        urllib.request.urlretrieve(data_url, data_file_path)
        print("Download complete.")

    train_data, val_data, chars, vocab_size, collection = load_data('data.txt')

    block_size = 256
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    model = GPTLanguageModel(vocab_size, block_size, n_head, n_layer, n_embd, n_head, dropout)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    m.load_state_dict(torch.load('weights.pth', map_location=device))
    m.eval()


    collections = json.load(open('chords.json', 'r'))
    for collection in tqdm(collections):
        for idx in range(10):
            song_name = "".join(collection['title'].split())
            file_name = song_name + "_" + str(idx)
            dir_name = "generated"
            inp = build_inputs(collection)

            encoded_input = encode(inp, chars)
            context = encoded_input.unsqueeze(0).to(device)

            output_indices = m.generate(context, max_new_tokens=500)
            output_indices_list = output_indices[0].tolist()

            try:
                decoded_output = format_chords(decode(output_indices_list, chars))

                print(decoded_output)       
                save_to_midi(decoded_output, save_path="generated/midi/{}.mid".format(file_name))
                midi_to_mp3(soundfont_path='soundfront.sf2', inpath='generated/midi/{}.mid'.format(file_name), outpath='generated/mp3/{}.mp3'.format(file_name))
            except IndexError:
                pass