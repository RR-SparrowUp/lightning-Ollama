from typing import Dict, List

class CaliforniaTopicTemplate:
    SYSTEM_PROMPT = """You are a knowledgeable assistant focused on California-related topics. Your expertise includes:
- California history and culture
- Geography and natural landmarks
- Cities and regions
- Economy and industries
- Politics and government
- Tourism and attractions
- Climate and environment
- Current events in California

Guidelines:
1. Keep responses focused on California-related aspects
2. If the user goes off-topic, acknowledge their question but guide them back to California-related discussions
3. Use relevant California examples when explaining concepts
4. Be informative but concise
5. When appropriate, suggest related California topics that might interest the user

Remember: Your primary goal is to maintain engaging conversations about California while keeping users on topic."""

    # Topic categories for redirecting off-topic conversations
    TOPIC_REDIRECTS: Dict[str, str] = {
        "weather": "Speaking of weather, did you know California has several unique climate zones? From Mediterranean climate in the coast to alpine conditions in the Sierra Nevada.",
        "food": "That reminds me of California's diverse culinary scene! From farm-to-table restaurants in San Francisco to food trucks in LA.",
        "technology": "That brings to mind Silicon Valley, California's tech hub! Companies like Apple and Google have transformed the state's economy.",
        "entertainment": "That makes me think of Hollywood and California's entertainment industry! Did you know the first movie studio in Hollywood opened in 1911?",
        "nature": "If you're interested in nature, California has some of the most diverse landscapes in the US, from the Redwood forests to Death Valley!",
        "default": "That's interesting! Speaking of California..."
    }

    @classmethod
    def format_prompt(cls, user_input: str) -> str:
        """Format the complete prompt with system message and user input"""
        return f"{cls.SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"

    @classmethod
    def get_redirection(cls, off_topic_category: str) -> str:
        """Get appropriate redirection for off-topic conversations"""
        return cls.TOPIC_REDIRECTS.get(off_topic_category, cls.TOPIC_REDIRECTS["default"])

    @classmethod
    def create_chat_messages(cls) -> List[Dict[str, str]]:
        """Create initial chat messages with system prompt"""
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT}
        ]

    @classmethod
    def detect_off_topic(cls, message: str) -> bool:
        """Simple check if the message might be off-topic"""
        california_keywords = {
            'california', 'ca', 'sacramento', 'san francisco', 'los angeles', 'san diego',
            'silicon valley', 'hollywood', 'yosemite', 'death valley', 'pacific coast',
            'golden gate', 'sierra nevada', 'bay area', 'socal', 'norcal'
        }
        message_words = set(message.lower().split())
        return not any(keyword in message.lower() for keyword in california_keywords) 