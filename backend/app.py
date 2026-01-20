from flask import Flask, request, jsonify
from flask_cors import CORS
from gtts import gTTS
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import os
import json
import base64
import time

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model cache to avoid reloading
MODEL_CACHE = {}

# ============ CUSTOM MODEL CLASSES ============
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.enc_att = nn.Linear(enc_dim, att_dim)
        self.dec_att = nn.Linear(dec_dim, att_dim)
        self.full_att = nn.Linear(att_dim, 1)
    
    def forward(self, enc, dec):
        att = self.full_att(torch.relu(self.enc_att(enc) + self.dec_att(dec).unsqueeze(1)))
        alpha = torch.softmax(att, dim=1)
        return (enc * alpha).sum(dim=1), alpha

class Decoder(nn.Module):
    def __init__(self, att_dim, emb_dim, dec_dim, vocab_size, enc_dim=2048, drop=0.5):
        super().__init__()
        self.enc_dim = enc_dim
        self.vocab_size = vocab_size
        self.attention = Attention(enc_dim, dec_dim, att_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        self.fc = nn.Linear(dec_dim, vocab_size)

# Image transform for custom model
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model_and_processor(model_name):
    if model_name not in MODEL_CACHE:
        if model_name == 'nic':
            print("Loading NIC Model...")
            model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            MODEL_CACHE['nic'] = (model, processor, tokenizer)
        elif model_name == 'blip':
            print("Loading BLIP Model...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            MODEL_CACHE['blip'] = (model, processor)
        elif model_name == 'custom':
            print("Loading Custom Model...")
            model_path = os.path.join(os.path.dirname(__file__), 'custom_caption_model.pth')
            vocab_path = os.path.join(os.path.dirname(__file__), 'vocab.json')
            
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                raise FileNotFoundError("Custom model files not found. Please train the model first.")
            
            # Load vocab
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            word2idx = vocab_data['word2idx']
            idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # Load ResNet152 encoder
            resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
            encoder = nn.Sequential(*list(resnet.children())[:-2]).to(DEVICE).eval()
            
            # Load decoder
            decoder = Decoder(
                checkpoint['attention_dim'],
                checkpoint['embed_dim'],
                checkpoint['decoder_dim'],
                checkpoint['vocab_size'],
                checkpoint['encoder_dim']
            ).to(DEVICE)
            decoder.load_state_dict(checkpoint['model_state_dict'])
            decoder.eval()
            
            MODEL_CACHE['custom'] = (encoder, decoder, word2idx, idx2word)
            
    return MODEL_CACHE[model_name]

def generate_custom_caption(image, encoder, decoder, word2idx, idx2word, max_len=20, beam_size=3):
    """Generate caption using beam search for custom model."""
    img = custom_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = encoder(img)  # (1, 2048, 7, 7)
        enc_out = features.permute(0, 2, 3, 1).view(1, -1, 2048)
        
        mean = enc_out.mean(1)
        h = decoder.init_h(mean)
        c = decoder.init_c(mean)
        
        start_idx = word2idx["<start>"]
        end_idx = word2idx["<end>"]
        
        candidates = [(0.0, [start_idx], h, c)]
        
        for _ in range(max_len):
            all_cands = []
            for score, seq, h_prev, c_prev in candidates:
                if seq[-1] == end_idx:
                    all_cands.append((score, seq, h_prev, c_prev))
                    continue
                
                emb = decoder.embedding(torch.tensor([seq[-1]]).to(DEVICE))
                ctx, _ = decoder.attention(enc_out, h_prev)
                h_new, c_new = decoder.lstm(torch.cat([emb, ctx], 1), (h_prev, c_prev))
                logits = decoder.fc(h_new)
                log_probs = F.log_softmax(logits, dim=1)
                
                top_probs, top_idx = log_probs.topk(beam_size)
                for i in range(beam_size):
                    all_cands.append((score + top_probs[0][i].item(), seq + [top_idx[0][i].item()], h_new, c_new))
            
            candidates = sorted(all_cands, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(c[1][-1] == end_idx for c in candidates):
                break
        
        best = candidates[0][1]
        return " ".join([idx2word[i] for i in best if i not in {start_idx, end_idx}])

@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    model_type = request.form.get('model', 'placeholder')

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    caption_text = ""

    try:
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        
        if model_type == 'nic':
            model, processor, tokenizer = get_model_and_processor('nic')
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            caption_text = preds[0].strip()
        
        elif model_type == 'blip':
            model, processor = get_model_and_processor('blip')
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption_text = processor.decode(out[0], skip_special_tokens=True)
        
        elif model_type == 'custom':
            encoder, decoder, word2idx, idx2word = get_model_and_processor('custom')
            caption_text = generate_custom_caption(image, encoder, decoder, word2idx, idx2word)
        
        else:
            return jsonify({'error': f'Unknown model: {model_type}'}), 400

    except Exception as e:
        print(f"Error generating caption: {e}")
        return jsonify({'error': str(e)}), 500
    
    # Generate Audio
    tts = gTTS(text=caption_text, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')

    return jsonify({
        'caption': caption_text,
        'audio': audio_base64
    })

if __name__ == '__main__':
    print("Preloading models... This may take a moment.")
    get_model_and_processor('nic')
    get_model_and_processor('blip')
    # Try to load custom model if it exists
    try:
        get_model_and_processor('custom')
    except FileNotFoundError:
        print("Custom model not found, skipping preload.")
    print("Models preloaded successfully.")
    
    app.run(debug=True, port=5000)

