from flask import Flask, render_template, request, jsonify
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import subprocess
import os
import time
from chat_template import CaliforniaTopicTemplate
from typing import List
import threading
import queue
import re

app = Flask(__name__)

# Add this global variable for tracking download progress
download_progress = {}

def pull_model(model_name):
    """Pull the specified model using ollama"""
    print(f"Pulling model {model_name}...")
    try:
        subprocess.run(
            ['docker', 'exec', '-it', 'ollama', 'ollama', 'pull', model_name],
            check=True,
            text=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")
        return False

def check_model_exists(model_name):
    """Check if the model exists in ollama"""
    try:
        result = subprocess.run(
            ['docker', 'exec', 'ollama', 'ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        return model_name in result.stdout
    except subprocess.CalledProcessError:
        return False

# Initialize Ollama with LangChain
model_name = os.getenv('OLLAMA_MODEL', 'llama2')  # Default to llama2 if not specified

def initialize_model():
    """Initialize the LLM and conversation chain"""
    global llm, conversation
    
    # Ensure model is available
    if not check_model_exists(model_name):
        if not pull_model(model_name):
            raise Exception(f"Failed to pull model {model_name}")
        time.sleep(2)
    
    llm = Ollama(
        model=model_name,
        base_url="http://localhost:11434"
    )
    
    template = """
{system_prompt}

You are an enthusiastic California expert. Follow these guidelines:

1. For California topics:
   - Share detailed, accurate information enthusiastically
   - Never apologize - you're meant to talk about California!
   - Include specific facts and interesting details
   - Encourage follow-up questions about the topic

2. For non-California topics:
   - Smoothly transition to a related California topic
   - Share interesting California facts that relate to their interest
   - Be natural and conversational

Remember: You're an expert who loves sharing California knowledge. No need to apologize for staying on topic - that's your specialty!

Previous conversation:
{history}

Current question: {input}
Assistant: """

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template,
        partial_variables={"system_prompt": CaliforniaTopicTemplate.SYSTEM_PROMPT}
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(
            ai_prefix="Assistant",
            human_prefix="Human",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        ),
        prompt=prompt,
        verbose=True
    )

@app.route('/')
def home():
    """Main landing page"""
    return render_template('Home.html')  # Changed from index.html to Home.html

@app.route('/chat')
def chat_interface():
    """Chat interface page"""
    return render_template('index.html')

@app.route('/chat_message', methods=['POST'])  # Renamed from /chat to /chat_message
def chat_message():
    """Handle chat messages"""
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Clean the message
        clean_message = user_message.replace('CALIFORNIA_TOPIC: ', '')
        
        # Get response from the model
        response = conversation.predict(input=clean_message)
        
        # Clean up repetitive introductions
        response = clean_response(response)
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

def clean_response(response: str) -> str:
    """Clean up repetitive parts of the response"""
    # Remove common repetitive introductions
    intros_to_remove = [
        "Hello there! As a knowledgeable assistant focused solely on California-related topics,",
        "Of course! As a knowledgeable assistant focused solely on California-related topics,",
        "I'm here to help with any questions or topics you'd like to learn more about.",
        "I'd be happy to help with any questions you may have about California.",
        "Of course! I'd be happy to help!",
        "Please let me know how I can assist you.",
        "Hi there!",
        "Hello!",
    ]
    
    cleaned = response
    for intro in intros_to_remove:
        cleaned = cleaned.replace(intro, "")
    
    # Clean up any double spaces and trim
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip()
    
    return cleaned

def check_ollama_ready():
    """Check if Ollama service is ready"""
    max_retries = 5
    for i in range(max_retries):
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True
        except:
            pass
        print(f"Waiting for Ollama to be ready... ({i+1}/{max_retries})")
        time.sleep(2)
    return False

@app.route('/home')
def homepage():
    return render_template('Home.html')

@app.route('/download_model', methods=['POST'])
def download_model():
    try:
        model_name = request.json.get('model')
        if not model_name:
            return jsonify({'success': False, 'error': 'No model specified'}), 400
        
        # Initialize progress tracking for this model
        download_progress[model_name] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Starting download...'
        }
        
        # Start download in background thread
        thread = threading.Thread(
            target=download_model_with_progress,
            args=(model_name,)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Download started',
            'model': model_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def download_model_with_progress(model_name):
    """Download model and track progress"""
    try:
        # Start the download process
        process = subprocess.Popen(
            ['docker', 'exec', 'ollama', 'ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Update progress based on output
        for line in process.stdout:
            if 'pulling manifest' in line.lower():
                download_progress[model_name] = {
                    'status': 'downloading',
                    'progress': 10,
                    'message': 'Pulling manifest...'
                }
            elif 'downloading' in line.lower():
                # Extract progress percentage if available
                match = re.search(r'(\d+)%', line)
                if match:
                    progress = int(match.group(1))
                    download_progress[model_name] = {
                        'status': 'downloading',
                        'progress': progress,
                        'message': f'Downloading... {progress}%'
                    }
            elif 'verifying' in line.lower():
                download_progress[model_name] = {
                    'status': 'verifying',
                    'progress': 95,
                    'message': 'Verifying download...'
                }
        
        # Check if download was successful
        if process.wait() == 0:
            download_progress[model_name] = {
                'status': 'completed',
                'progress': 100,
                'message': 'Download completed successfully'
            }
        else:
            download_progress[model_name] = {
                'status': 'error',
                'progress': 0,
                'message': 'Download failed'
            }
            
    except Exception as e:
        download_progress[model_name] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

@app.route('/download_progress/<model_name>')
def get_download_progress(model_name):
    """Get current download progress for a model"""
    if model_name not in download_progress:
        return jsonify({
            'status': 'not_found',
            'progress': 0,
            'message': 'Download not started'
        })
    
    return jsonify(download_progress[model_name])

@app.route('/list_models')
def list_models():
    try:
        result = subprocess.run(
            ['docker', 'exec', 'ollama', 'ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to get model information
        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line:
                parts = line.split()
                models.append({
                    'name': parts[0],
                    'size': parts[1],
                    'modified': ' '.join(parts[2:])
                })
                
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_model', methods=['POST'])
def delete_model():
    try:
        model_name = request.json.get('model')
        if not model_name:
            return jsonify({'success': False, 'error': 'No model specified'}), 400
            
        result = subprocess.run(
            ['docker', 'exec', 'ollama', 'ollama', 'rm', model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Wait for Ollama to be ready
    print("Checking Ollama service...")
    if not check_ollama_ready():
        print("Error: Ollama service is not responding")
        print("Please ensure Ollama container is running")
        exit(1)
    
    # Initialize the model
    try:
        print(f"Initializing model {model_name}...")
        initialize_model()
        print(f"Successfully initialized model: {model_name}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Please ensure Ollama is running and the model is available")
        exit(1)
    
    print(f"Starting web interface on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 