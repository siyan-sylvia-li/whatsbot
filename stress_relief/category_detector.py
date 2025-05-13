import re
from typing import Tuple
from openai import OpenAI


class CategoryDetector:
    """Detects stress categories from user messages using LLM."""
    
    def __init__(self, llm_client=None, openai_api_key=None):
        """Initialize the CategoryDetector.
        
        Args:
            llm_client: Optional custom LLM client for category detection.
            openai_api_key: Optional OpenAI API key. If provided, OpenAI's API will be used.
            
        Raises:
            ValueError: If neither llm_client nor openai_api_key is provided.
        """
        self.llm_client = llm_client
        self.openai_api_key = openai_api_key
        
        # Set up OpenAI client if API key is provided
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            
        # Ensure we have LLM capabilities
        if not (self.llm_client or self.openai_client):
            raise ValueError("Either llm_client or openai_api_key must be provided")
    
    def detect_category(self, message: str) -> Tuple[str, float]:
        """Detect the stress category from a user message using LLM.
        
        Args:
            message: The user's message.
            
        Returns:
            A tuple of (category, confidence).
            
        Raises:
            RuntimeError: If LLM detection fails.
        """
        try:
            return self._llm_detect_category(message)
        except Exception as e:
            raise RuntimeError(f"LLM category detection failed: {e}")
    
    def _llm_detect_category(self, message: str) -> Tuple[str, float]:
        """Use LLM to detect the stress category.
        
        Args:
            message: The user's message.
            
        Returns:
            A tuple of (category, confidence).
            
        Raises:
            Exception: If LLM detection fails.
        """
        # Define the categories we want to detect
        categories = ["anxiety", "sadness", "anger", "overwhelm", "general"]
        
        # Set up the prompt for the LLM
        prompt = f"""
        You are a stress detection system. Analyze the following message and determine the primary type of stress the person is experiencing.
        
        Categories:
        - anxiety: Nervousness, worry, fear, panic
        - sadness: Feeling down, depressed, unhappy, grieving
        - anger: Frustrated, mad, irritated, annoyed, upset
        - overwhelm: Too much to handle, burnout, exhausted
        - general: General stress or tension without a specific category
        
        User message: "{message}"
        
        Respond in the following format ONLY:
        CATEGORY: [category name]
        CONFIDENCE: [confidence score from 0.0 to 1.0]
        
        For example:
        CATEGORY: anxiety
        CONFIDENCE: 0.8
        """
        
        # If we have a custom LLM client, use it
        if self.llm_client:
            raw_response = self.llm_client.generate(prompt)
        
        # Otherwise, use OpenAI
        else:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a stress detection AI that analyzes text and categorizes the type of stress."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.0  # Use low temperature for more deterministic responses
            )
            raw_response = response.choices[0].message.content
        
        # Parse the LLM response to extract category and confidence
        # Extract category and confidence using regex
        category_match = re.search(r'CATEGORY:\s*(\w+)', raw_response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', raw_response, re.IGNORECASE)
        
        if category_match and confidence_match:
            category = category_match.group(1).lower()
            confidence = float(confidence_match.group(1))
            
            # Validate the category
            if category in categories:
                return category, confidence
        
        # If parsing failed or invalid category, try a simpler approach
        lines = raw_response.strip().split('\n')
        for line in lines:
            if line.lower().startswith('category:'):
                category = line.split(':', 1)[1].strip().lower()
                if category in categories:
                    # If we found a valid category but no confidence, assign a default
                    confidence = 0.7
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                    return category, confidence
        
        # If all parsing attempts fail, raise an exception
        raise ValueError(f"Could not parse LLM response: {raw_response}") 