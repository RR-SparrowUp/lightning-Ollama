from flask import Flask, render_template, request, jsonify
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

app = Flask(__name__)

# Initialize Ollama with LangChain
model_name = os.getenv('OLLAMA_MODEL', 'llama2')  # Default to llama2 if not specified
llm = Ollama(model=model_name, base_url="http://localhost:11434")
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
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

        # Get response from the model
        response = conversation.predict(input=user_message)
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Ensure Ollama is running and accessible
    try:
        llm.invoke("test")
        print(f"Successfully connected to Ollama model: {model_name}")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running and accessible at http://localhost:11434")
        exit(1)
    
    print("Starting web interface on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 