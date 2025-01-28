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
    
    # Create a more conversational prompt template
    template = """
{system_prompt}

IMPORTANT: You are a California specialist. Follow these rules:

1. For non-California topics:
   - Politely redirect to a relevant California topic
   - Provide specific California information as an alternative
   Example: "What's the weather in New York?"
   Response: "While I focus on California, I can tell you about our diverse climate zones! San Francisco has unique microclimates, while Los Angeles enjoys Mediterranean weather. Which California region interests you?"

2. For California topics:
   - Respond enthusiastically and informatively
   - Never apologize when discussing California topics
   - Provide specific details and interesting facts
   Example: "Tell me about Death Valley"
   Response: "Death Valley is one of California's most fascinating places! It holds the record for the highest temperature ever recorded (134°F) and features stunning landscapes from salt flats to sand dunes. Would you like to know about its unique geology, wildlife, or best times to visit?"

Previous conversation:
{history}

Current question: {input}
Assistant: """

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template,
        partial_variables={"system_prompt": CaliforniaTopicTemplate.SYSTEM_PROMPT}
    )
    
    # Initialize conversation with better memory handling
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

        # Clean the message
        clean_message = user_message.replace('CALIFORNIA_TOPIC: ', '')
        
        # Get conversation history
        history = conversation.memory.chat_memory.messages
        
        # Check if it's off-topic but related to previous California discussion
        if any(prev_msg.content and 'california' in prev_msg.content.lower() 
               for prev_msg in history[-4:]):  # Check last 2 exchanges
            # Try to maintain context while redirecting
            clean_message = f"Continuing our discussion about California, regarding {clean_message}"
        
        # Get response from the model
        response = conversation.predict(input=clean_message)
        
        # Clean up and humanize the response
        response = humanize_response(response, history)
        
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

def humanize_response(response: str, history: List) -> str:
    """Make responses more natural and contextual"""
    # Remove common repetitive phrases
    response = clean_response(response)
    
    # Check if the topic is about California
    california_keywords = {
        'death valley', 'yosemite', 'san francisco', 'los angeles', 'sacramento',
        'silicon valley', 'hollywood', 'california', 'san diego', 'sequoia',
        'redwood', 'napa', 'lake tahoe', 'joshua tree', 'big sur'
    }
    
    user_message = history[-1].content.lower() if history else ""
    is_california_topic = any(keyword in user_message.lower() for keyword in california_keywords)
    
    # If it's a California topic, remove any apologetic phrases
    if is_california_topic:
        response = response.replace("I apologize, but ", "")
        response = response.replace("I'm sorry, but ", "")
        response = response.replace("I can only discuss California-related topics. ", "")
        
        # If the response became too generic after cleaning
        if len(response) < 50 or "would you like to know" in response.lower():
            return generate_california_response(user_message)
    else:
        # For non-California topics, use the location alternative
        if any(place in user_message for place in ['new york', 'chicago', 'london', 'tokyo']):
            return generate_location_alternative(user_message)
        elif 'food' in user_message:
            return "California has an incredible food scene! From fresh seafood in San Francisco to diverse ethnic cuisines in Los Angeles. Would you like to explore California's culinary highlights?"
        elif 'weather' in user_message:
            return "California has fascinating weather patterns! From Mediterranean climate in the coast to alpine conditions in the mountains. Would you like to learn about a specific region's weather?"
    
    return response

def generate_location_alternative(user_message: str) -> str:
    """Generate location-specific California alternatives"""
    if 'weather' in user_message:
        return "Let me tell you about California's diverse climate zones instead! We have everything from Mediterranean coastal weather to alpine mountain conditions. Which region interests you?"
    elif 'food' in user_message:
        return "While I focus on California, I can tell you about our amazing food scene! From famous In-N-Out burgers to Michelin-starred restaurants in San Francisco. What type of California cuisine interests you?"
    elif 'city' in user_message or 'downtown' in user_message:
        return "I'd love to tell you about California's vibrant cities instead! From tech-hub San Francisco to entertainment capital Los Angeles. Which California city would you like to explore?"
    else:
        return "As a California specialist, I can tell you about similar aspects here! We have world-class cities, unique culture, amazing nature, and more. What part of California interests you?"

def generate_california_response(topic: str) -> str:
    """Generate enthusiastic responses for California topics"""
    california_topics = {
        'death valley': """Death Valley is one of California's most extraordinary places! This vast desert landscape holds the record for the highest temperature ever recorded (134°F) and sits at the lowest elevation in North America. Its stunning features include the mysterious moving rocks at Racetrack Playa, colorful Artist's Palette, and towering sand dunes. What aspect of Death Valley interests you most?""",
        
        'yosemite': """Yosemite is a crown jewel of California's national parks! Home to iconic granite cliffs like El Capitan and Half Dome, thundering waterfalls, ancient giant sequoias, and diverse wildlife. Would you like to know about its natural wonders, hiking trails, or best times to visit?""",
        
        'san francisco': """San Francisco is a vibrant city known for its iconic Golden Gate Bridge, historic cable cars, and diverse neighborhoods! The city offers everything from world-class restaurants to cultural attractions, tech innovation, and beautiful bay views. What would you like to know about SF?"""
        # Add more California topics as needed
    }
    
    # Find the most relevant topic
    for key, response in california_topics.items():
        if key in topic.lower():
            return response
    
    return response

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