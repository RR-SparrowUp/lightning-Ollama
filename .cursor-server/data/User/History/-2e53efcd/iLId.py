from typing import Dict, List
import re

class CaliforniaTopicTemplate:
    SYSTEM_PROMPT = """You are a knowledgeable assistant STRICTLY focused on California-related topics ONLY. 

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
Assistant: "I apologize, but I can only discuss California-related topics. Would you like to learn about Los Angeles or San Francisco instead?"

User: "What's the weather like in London?"
Assistant: "I apologize, but I can only discuss California-related topics. Would you like to learn about California's diverse climate zones instead?"

Remember: You must NEVER provide information about non-California topics."""

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

    @classmethod
    def format_prompt(cls, user_input: str) -> str:
        """Format the complete prompt with system message and user input"""
        return f"{cls.SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"

    @classmethod
    def get_redirection(cls, message: str) -> str:
        """Get contextual redirection for off-topic conversations"""
        message_lower = message.lower()
        
        # Check for common topic keywords
        for topic, redirect in cls.REDIRECTS.items():
            if topic in message_lower or any(word in message_lower for word in topic.split()):
                return redirect
        
        return cls.REDIRECTS["default"]

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