from typing import Dict, List
import re
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import geopy 
from Countrydetails import countries 

class CaliforniaTopicTemplate:
    SYSTEM_PROMPT = f"""You are a knowledgeable assistant STRICTLY focused on California-related topics ONLY. 

Your expertise includes:
- California history and culture
- Geography and natural landmarks
- Cities and regions
- Economy and industries
- Politics and government
- Tourism and attractions
- Climate and environment
- Current events in California

STRICT GUIDELINES:
1. ONLY answer questions about California
2. If a user asks about non-California topics, ALWAYS respond with:
   "I apologize, but I can only discuss California-related topics. Would you like to know about [suggest relevant California topic]?"
3. Never provide information about other states or countries
4. Keep all examples and analogies California-specific
5. Redirect all off-topic questions back to California

Example responses:
User: "Tell me about New York"
Assistant: "I apologize, but I can only discuss California-related topics. Would you like to learn something about California ?" 

User: "What's the weather like in London?"
Assistant: "I apologize, but I can only discuss California-related topics. Would you like to learn about California ?"

Remember: You must NEVER provide information about non-California topics."""

    ## use topic modelling to understand best close topic to be suggested based on the known topics?
    # Topic categories for redirecting off-topic conversations
    REDIRECTS: Dict[str, str] = {
        "cities": "Would you like to learn about California's major cities like Los Angeles, San Francisco, or San Diego instead?",
        "weather": "Would you like to learn about California's unique climate zones instead?",
        "food": "Would you like to explore California's diverse culinary scene, from farm-to-table restaurants to food trucks?",
        "technology": "Would you like to learn about Silicon Valley, California's world-famous tech hub?",
        "entertainment": "Would you like to learn about Hollywood and California's entertainment industry?",
        "nature": "Would you like to explore California's diverse landscapes, from Redwood forests to Death Valley?",
        "education": "Would you like to learn about California's prestigious universities like UC Berkeley, Stanford, or UCLA?",
        "default": "I apologize, but I can only discuss California-related topics. What aspect of California would you like to explore?"
    }

    # Initialize BERT model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
    except OSError:
        import subprocess
        subprocess.run(['pip', 'install', 'transformers', 'torch'])
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')

    # Define comprehensive topic hierarchies with BERT-friendly descriptions
    TOPIC_HIERARCHY = {
        'cities': {
            'description': 'Urban areas, city life, and metropolitan regions in California',
            'keywords': ['city', 'urban', 'metropolitan', 'downtown', 'population'],
            'examples': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
            'related_concepts': ['transportation', 'housing', 'development', 'community'],
            'subtopics': ['urban planning', 'city culture', 'local government']
        },
        'technology': {
            'description': 'Technology industry, innovation, and digital advancement in California',
            'keywords': ['tech', 'innovation', 'software', 'startup', 'digital'],
            'examples': ['Silicon Valley', 'Tech startups', 'Innovation hubs'],
            'related_concepts': ['entrepreneurship', 'venture capital', 'computing'],
            'subtopics': ['artificial intelligence', 'biotechnology', 'clean tech']
        },
        'entertainment': {
            'keywords': ['hollywood', 'movies', 'film', 'television', 'media'],
            'examples': ['Hollywood film industry', 'Entertainment studios', 'Media production'],
            'related_concepts': ['acting', 'production', 'streaming', 'cinema']
        },
        'nature': {
            'keywords': ['parks', 'wilderness', 'outdoors', 'wildlife', 'conservation'],
            'examples': ['Yosemite', 'Redwood forests', 'Death Valley'],
            'related_concepts': ['ecology', 'preservation', 'hiking', 'camping']
        },
        'climate': {
            'keywords': ['weather', 'temperature', 'climate', 'seasons', 'environment'],
            'examples': ['Mediterranean climate', 'Microclimates', 'Weather patterns'],
            'related_concepts': ['meteorology', 'precipitation', 'humidity', 'forecast']
        }
        
    }

    @classmethod
    def get_bert_embedding(cls, text: str) -> np.ndarray:
        """Get BERT embedding for text"""
        # Tokenize and prepare input
        inputs = cls.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = cls.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings[0]  # Return the first (and only) embedding

    @classmethod
    def calculate_topic_similarity(cls, message: str) -> List[tuple]:
        """Calculate similarity between message and topics using BERT"""
        message_embedding = cls.get_bert_embedding(message)
        
        similarities = []
        for topic, details in cls.TOPIC_HIERARCHY.items():
            # Create comprehensive topic description
            topic_text = f"{details['description']} {' '.join(details['keywords'])} {' '.join(details['examples'])}"
            topic_embedding = cls.get_bert_embedding(topic_text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                message_embedding.reshape(1, -1),
                topic_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append((topic, similarity, details))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)

    @classmethod
    def find_related_topics(cls, message: str, num_topics: int = 2) -> List[str]:
        """Find most relevant topics using BERT embeddings"""
        # Get topic similarities
        topic_similarities = cls.calculate_topic_similarity(message)
        
        # Get top N most relevant topics
        suggestions = []
        for topic, similarity, details in topic_similarities[:num_topics]:
            if similarity > 0.4:  # Adjusted threshold for BERT
                # Choose example based on subtopic relevance
                relevant_example = np.random.choice(details['examples'])
                relevant_subtopic = np.random.choice(details['subtopics'])
                suggestions.append(f"{relevant_example} ({relevant_subtopic})")
        
        if not suggestions:
            # Fallback to most similar topic
            topic, _, details = topic_similarities[0]
            return [f"{np.random.choice(details['examples'])} ({topic})"]
        
        return suggestions

    @classmethod
    def get_redirection(cls, message: str) -> str:
        """Get contextual redirection using BERT-based similarity"""
        related_topics = cls.find_related_topics(message)
        
        if len(related_topics) > 1:
            topics_str = f"{', '.join(related_topics[:-1])} or {related_topics[-1]}"
        else:
            topics_str = related_topics[0]
            
        return (f"I apologize, but I can only discuss California-related topics. "
                f"Based on your interest, would you like to learn about {topics_str}?")

    @classmethod
    def format_prompt(cls, user_input: str) -> str:
        """Format the complete prompt with system message and user input"""
        return f"{cls.SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"

    @classmethod
    def create_chat_messages(cls) -> List[Dict[str, str]]:
        """Create initial chat messages with system prompt"""
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT}
        ]

    @classmethod
    def detect_off_topic(cls, message: str) -> bool:
        """Enhanced check if the message might be off-topic"""
        # List of US states (excluding California) and major world cities/countries
        non_california_locations = {
            'new york', 'texas', 'florida', 'chicago', 'miami', 'london', 
            'paris', 'tokyo', 'seattle', 'boston', 'vegas', 'hawaii',
            'china', 'india', 'europe', 'africa', 'australia'
        }
        
        message_lower = message.lower()
        
        # If message contains non-California locations, it's off-topic
        if any(location in message_lower for location in non_california_locations):
            return True
            
        # California-related keywords
        california_keywords = {
            # Regions and cities
            'california', 'ca', 'sacramento', 'san francisco', 'los angeles', 'la', 
            'san diego', 'silicon valley', 'hollywood', 'bay area', 'socal', 'norcal',
            'oakland', 'berkeley', 'palo alto', 'san jose', 'fresno',
            
            # Landmarks and nature
            'yosemite', 'death valley', 'pacific coast', 'golden gate', 'sierra nevada',
            'redwood', 'sequoia', 'mojave', 'lake tahoe', 'big sur',
            
            # Institutions
            'ucla', 'usc', 'stanford', 'berkeley', 'caltech',
            
            # Topics
            'earthquake', 'tech', 'wine country', 'napa', 'sonoma'
        }
        
        # Check for California-related content
        has_california_content = any(keyword in message_lower for keyword in california_keywords)
        
        # If it's a greeting or very short message, don't consider it off-topic
        if len(message.split()) <= 3 and any(word in message_lower for word in {'hi', 'hello', 'hey', 'help'}):
            return False
            
        return not has_california_content 