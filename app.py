from flask import Flask, request, jsonify
import torch
import os
import joblib
from model import RobertaClassifier
from transformers import RobertaTokenizer
from dataset import PredictionData
from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_LEN = 350
app = Flask(__name__)

#First we load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('pytorch_roberta_sentiment.bin', map_location=device)
model.to(device)
model.eval()
#Then we load the label encoder
class_encoder = joblib.load('class_encoder.pkl')
#Then we load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)


# Get BATCH_SIZE from environment variable, default to 32 if not set
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))

# Function to be called for single prediction
def _predict(model, input_text, tokenizer, class_encoder):
    text = str(input_text)
    text = " ".join(text.split()) #Remove extra and irregular white spaces
    inputs = tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length= MAX_LEN,
                truncation = True,
                padding = 'max_length',
                return_token_type_ids=True
            )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(device, dtype=torch.long)[None,:]
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device, dtype=torch.long)[None,:]
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).to(device, dtype=torch.long)[None,:]

    model.eval()
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
        big_val, big_idx = torch.max(outputs.data, dim=1) 
        prediction = big_idx.item()

    return prediction, class_encoder.inverse_transform([prediction])[0]

#Function to be called for batched prediction
def _batched_predict(model, input_texts, tokenizer, class_encoder):
    prediction_set = PredictionData(input_texts, tokenizer, MAX_LEN)
    prediction_loader = DataLoader(prediction_set, batch_size = BATCH_SIZE, num_workers=0)

    predicted_indices = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(prediction_loader)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)

            big_val, big_idx = torch.max(outputs.data, dim=1)

            big_idx = big_idx.cpu().numpy()

            predicted_indices.extend(list(big_idx))
            predicted_labels.extend(list(class_encoder.inverse_transform(big_idx)))


    return predicted_indices, predicted_labels



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', None)
    if text is None:
        return jsonify({'error': 'No text provided'}), 400
    _, prediction = _predict(model, text, tokenizer, class_encoder)
    return jsonify({'prediction': prediction})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    texts = data.get('texts', None)
    if texts is None or not isinstance(texts, list):
        return jsonify({'error': 'No list of texts provided'}), 400
    
    _, predictions = _batched_predict(model, texts, tokenizer, class_encoder)
    
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
