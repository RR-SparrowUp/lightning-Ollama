from flask import Flask, render_template, request, jsonify
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import subprocess
import os
import time
from chat_template import CaliforniaTopicTemplate

app = Flask(__name__)

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
    
    # Create a more focused prompt template
    template = """
{system_prompt}

Focus on providing direct, informative answers about California.
Avoid repetitive greetings and unnecessary acknowledgments.

Current conversation:
{history}
Human: {input}
Assistant: """

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template,
        partial_variables={"system_prompt": CaliforniaTopicTemplate.SYSTEM_PROMPT}
    )
    
    # Initialize conversation with California template
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(ai_prefix="Assistant", human_prefix="Human"),
        prompt=prompt,
        verbose=True
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_model')
def get_model():
    return jsonify({'model': model_name})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Check if message is off-topic
        if CaliforniaTopicTemplate.detect_off_topic(user_message):
            # Get contextual redirection
            redirection = CaliforniaTopicTemplate.get_redirection(user_message)
            return jsonify({'response': redirection})
        else:
            # For on-topic messages, remove the prefix marker
            clean_message = user_message.replace('CALIFORNIA_TOPIC: ', '')
            
            # Get response from the model
            response = conversation.predict(input=clean_message)
            
            # Clean up repetitive parts of the response
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
        "Please let me know how I can assist you."
    ]
    
    cleaned = response
    for intro in intros_to_remove:
        cleaned = cleaned.replace(intro, "")
    
    # Clean up any double spaces and trim
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip()
    
    # If the response is now too short, return a prompt for more specific questions
    if len(cleaned) < 50:
        return "Could you please ask a more specific question about California?"
    
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