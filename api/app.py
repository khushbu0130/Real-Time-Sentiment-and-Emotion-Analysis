from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from deepface import DeepFace
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import logging
import torch

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize models
try:
    sentiment_pipeline = pipeline('sentiment-analysis')
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        'j-hartmann/emotion-english-distilroberta-base')
    emotion_tokenizer = AutoTokenizer.from_pretrained(
        'j-hartmann/emotion-english-distilroberta-base')
    emotion_labels = emotion_model.config.id2label

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5)
    
    # Define the MPAA ratings
    ratings = ["G", "PG", "PG-13", "R", "NC-17"]
    
    app.logger.info("All models loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {e}")


@app.route('/')
def home():
    """Root endpoint to test if the server is running"""
    return jsonify({
        "message": "Content Analysis API is running",
        "endpoints": [
            "/api/data - GET",
            "/upload - POST", 
            "/analyze - POST",
            "/analyze-url - POST"
        ]
    })


@app.route('/api/data')
def get_data():
    return jsonify({"message": "API is running"})


def analyze_text(text):
    """Analyze text for sentiment and emotion"""
    if not text or not isinstance(text, str):
        app.logger.error("Invalid text input for analysis")
        return "Neutral", "Wonder"
    
    max_length = 512
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    sentiments = []
    emotions = []

    for chunk in chunks:
        if not chunk.strip():
            continue
            
        try:
            inputs = emotion_tokenizer(
                chunk, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                sentiment_result = sentiment_pipeline(chunk)
                emotion_result = emotion_model(**inputs)
                emotion_probs = torch.nn.functional.softmax(
                    emotion_result.logits, dim=-1)
                emotion_label = emotion_labels[torch.argmax(emotion_probs).item()]

            sentiments.extend(sentiment_result)
            emotions.append(emotion_label)
        except Exception as e:
            app.logger.error(f"Error analyzing chunk: {e}")
            continue

    return map_sentiment(sentiments), map_emotion(emotions)


def map_sentiment(sentiment_results):
    """Map sentiment results to standardized format"""
    if not sentiment_results:
        return "Neutral"
        
    sentiment_mapping = {
        'POSITIVE': 'Positive',
        'NEGATIVE': 'Negative',
        'NEUTRAL': 'Neutral'
    }
    
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for result in sentiment_results:
        sentiment = result['label'].upper()
        sentiment_counts[sentiment_mapping.get(sentiment, 'Neutral')] += 1
    
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return overall_sentiment


def map_emotion(emotion_results):
    """Map emotion results to standardized format"""
    if not emotion_results:
        return "Wonder"
        
    emotion_mapping = {
        'love': 'Romance',
        'joy': 'Humor',
        'compassion': 'Compassion',
        'anger': 'Rage',
        'valor': 'Valor',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'surprise': 'Wonder',
        'peace': 'Peace',
        'sadness': 'Sadness'
    }
    
    emotion_counts = {emotion: 0 for emotion in emotion_mapping.values()}
    emotion_counts['Wonder'] = 0  # Default category
    
    for emotion in emotion_results:
        mapped_emotion = emotion_mapping.get(emotion.lower(), 'Wonder')
        emotion_counts[mapped_emotion] += 1
    
    most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
    return most_frequent_emotion


def analyze_image_emotion(image_path):
    """Analyze emotion from image using DeepFace"""
    try:
        emotion_analysis = DeepFace.analyze(image_path, actions=['emotion'])
        dominant_emotion = emotion_analysis[0].get('dominant_emotion')
        return dominant_emotion
    except Exception as e:
        app.logger.error(f"Error in image emotion analysis: {str(e)}")
        return "neutral"


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        return ""


def extract_text_from_txt(txt_path):
    """Extract text from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        app.logger.error(f"Error extracting text from TXT: {e}")
        return ""


def extract_text_from_url(url):
    """Extract text from web URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            app.logger.error(f"Error fetching URL: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = ' '.join([para.get_text() for para in paragraphs])

        if not text.strip():
            app.logger.error("Extracted text is empty.")
            return None

        return text
    except Exception as e:
        app.logger.error(f"Error extracting text from URL: {e}")
        return None


def determine_age_appropriateness(text):
    """Determine MPAA rating based on text content"""
    try:
        inputs = tokenizer(text, return_tensors="pt",
                          padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return ratings[prediction]
    except Exception as e:
        app.logger.error(f"Error determining age appropriateness: {e}")
        return "PG"


def determine_mpaa_rating(emotion):
    """Determine MPAA rating based on emotion (heuristic)"""
    emotion_lower = emotion.lower()
    if emotion_lower in ['angry', 'anger', 'fear', 'disgust', 'sad', 'sadness']:
        return 'Negative', 'R'
    elif emotion_lower in ['happy', 'joy', 'surprise']:
        return 'Positive', 'G'
    else:
        return 'Neutral', 'PG'


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({'filename': filename, 'message': 'File uploaded successfully'}), 200
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'error': 'File upload failed'}), 500


@app.route('/analyze', methods=['POST'])
def analyze_content():
    """Analyze uploaded file content"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = {}
        
        # Handle image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
            emotion = analyze_image_emotion(file_path)
            sentiment, age_rating = determine_mpaa_rating(emotion)
            result = {
                'sentiment': sentiment,
                'emotion': emotion, 
                'age_rating': age_rating,
                'type': 'image'
            }

        # Handle text files
        elif filename.lower().endswith(('.pdf', '.txt')):
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            else:
                text = extract_text_from_txt(file_path)
                
            if not text or not isinstance(text, str):
                app.logger.error(f"Failed to extract text from file: {filename}")
                return jsonify({'error': 'Failed to extract text from file'}), 500
                
            app.logger.debug(f"Extracted text length: {len(text)}")
            sentiment, emotion = analyze_text(text)
            age_rating = determine_age_appropriateness(text)
            
            result = {
                'sentiment': sentiment,
                'emotion': emotion, 
                'age_rating': age_rating,
                'type': 'text',
                'text_length': len(text)
            }
        else:
            return jsonify({'error': 'Unsupported file type. Supported: PNG, JPG, JPEG, SVG, PDF, TXT'}), 400

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
            
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"Error analyzing content: {e}")
        return jsonify({'error': 'Content analysis failed', 'details': str(e)}), 500


@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze content from URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided in request body'}), 400

        url = data.get('url')
        if not url:
            return jsonify({'error': 'Empty URL provided'}), 400

        app.logger.debug(f"Fetching URL: {url}")
        text = extract_text_from_url(url)
        
        if not text or not isinstance(text, str):
            app.logger.error(f"Failed to extract valid text from URL: {url}")
            return jsonify({'error': 'Failed to extract text from URL'}), 500

        sentiment, emotion = analyze_text(text)
        age_rating = determine_age_appropriateness(text)
        
        result = {
            'sentiment': sentiment, 
            'emotion': emotion, 
            'age_rating': age_rating,
            'type': 'url',
            'text_length': len(text),
            'url': url
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"Error analyzing URL: {e}")
        return jsonify({'error': 'URL analysis failed', 'details': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)