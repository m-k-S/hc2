from flask import Flask, request, jsonify
import torch
from inference import load_model, load_vocab
from data import encode, decode
import os

app = Flask(__name__)


@app.before_request
def load_language_model():
    # The following line will remove this handler, making it
    # only run on the first request
    app.before_request_funcs[None].remove(load_language_model)

    train_data, val_data, chars, vocab_size, collection = load_vocab('data.txt')
    m = load_model('weights.pth', vocab_size)
    app.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.chars = chars
    app.model = m

@app.route('/generate', methods=['POST'])
def generate_music():
    content = request.json
    print (content)
    inp = content['input']
    length = content['length']
    encoded_input = encode(inp, app.chars)
    context = encoded_input.unsqueeze(0).to(app.device)

    output_indices = app.model.generate(context, max_new_tokens=int(length))
    output_indices_list = output_indices[0].tolist()

    try:
        decoded_output = decode(output_indices_list, app.chars)
        return jsonify({"status": "success", "message": decoded_output})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    data_file_path = 'data.txt'
    if not os.path.exists(data_file_path):
        import urllib.request
        print("Downloading data.txt...")
        data_url = 'https://www.dropbox.com/scl/fi/2yod9hatuawm8vjd8aox7/data.txt?rlkey=v530uyk0nrpuz8b66nzc41tmo&dl=1'  # Replace this URL with the actual URL
        urllib.request.urlretrieve(data_url, data_file_path)
        print("Download complete.")

    app.run(host="0.0.0.0", debug=True)
