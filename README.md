# Image Caption Generator

An AI-powered image captioning application that generates natural language descriptions for images using multiple deep learning models. The application also provides text-to-speech functionality to read the generated captions aloud.

## Features

- **Multiple Model Support**: Choose between three different captioning models
- **Text-to-Speech**: Automatically converts captions to audio
- **Modern Web Interface**: React-based frontend with a clean user experience

## Project Structure

```
image_caption/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── custom_caption_model.pth  # Trained custom model weights
│   └── vocab.json                # Custom model vocabulary
├── frontend/                     # React + Vite frontend
│   ├── src/
│   └── package.json
├── train_custom_model.ipynb      # Jupyter notebook for training custom model
└── evaluate_models.ipynb         # Jupyter notebook for model evaluation
```

## Models & Datasets

| Model | Architecture | Dataset |
|-------|--------------|---------|
| **NIC** | ViT + GPT-2 (`nlpconnect/vit-gpt2-image-captioning`) | [COCO](https://cocodataset.org/#download) |
| **BLIP** | Salesforce BLIP (`Salesforce/blip-image-captioning-base`) | [COCO](https://cocodataset.org/#download) |
| **Custom** | ResNet152 + LSTM + Attention | [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) |

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd image_caption/backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install flask flask-cors gtts transformers torch torchvision pillow
   ```

4. Run the backend server:
   ```bash
   python app.py
   ```
   The API will start at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd image_caption/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will start at `http://localhost:5173`

## Usage

1. Start both the backend and frontend servers
2. Open your browser and navigate to `http://localhost:5173`
3. Upload an image
4. Select a captioning model (NIC, BLIP, or Custom)
5. Click generate to get your caption with audio playback
