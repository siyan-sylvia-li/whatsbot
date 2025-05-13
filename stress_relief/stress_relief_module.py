import time
import random
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI

from .user_profile_manager import UserProfileManager
from .category_detector import CategoryDetector
from .intervention_database import InterventionDatabase


class StressReliefModule:
    """A prompt-based stress relief module with enhanced personalization using LLM."""
    
    def __init__(self, 
                 profile_path: str = "user_profiles.json", 
                 llm_client=None,
                 openai_api_key=None,
                 favorite_threshold: float = 7.5):
        """Initialize the StressReliefModule.
        
        Args:
            profile_path: Path to the JSON file where profiles will be stored.
            llm_client: Optional custom LLM client for more personalized responses.
            openai_api_key: Optional OpenAI API key for LLM-based responses.
            favorite_threshold: Threshold for considering an intervention a favorite.
            
        Raises:
            ValueError: If neither llm_client nor openai_api_key is provided.
        """
        self.profile_manager = UserProfileManager(profile_path)
        self.intervention_db = InterventionDatabase()
        self.llm_client = llm_client
        self.openai_api_key = openai_api_key
        self.favorite_threshold = favorite_threshold
        
        # Set up OpenAI client if API key is provided
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Ensure we have LLM capabilities
        if not (self.llm_client or self.openai_client):
            raise ValueError("Either llm_client or openai_api_key must be provided")
            
        # Initialize the category detector with LLM capabilities
        self.category_detector = CategoryDetector(llm_client, openai_api_key)
        
        # Track conversation state
        self.current_state = {}
        
        # Import needed modules
        self.time = time
        self.random = random
    
    def process_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """Process a user's message and return a response.
        
        Args:
            message: The user's message.
            user_id: The user's ID.
            
        Returns:
            A dictionary containing the response and metadata.
        """
        # Skip feedback check and directly process as a new message
        return self.process_user_message(message, user_id)
    
    def _check_for_feedback(self, message: str) -> Optional[int]:
        """Check if a message contains feedback using LLM.
        
        Args:
            message: The user's message.
            
        Returns:
            The feedback score (1-10) if found, None otherwise.
        """
        try:
            return self._llm_detect_feedback(message)
        except Exception as e:
            print(f"Error detecting feedback: {e}")
            return None
    
    def _llm_detect_feedback(self, message: str) -> Optional[int]:
        """Use LLM to detect if a message contains feedback and extract the score.
        
        Args:
            message: The user's message to analyze.
            
        Returns:
            The feedback score (1-10) if detected, None otherwise.
        """
        # Set up the prompt for the LLM
        prompt = f"""
        Analyze if the following message contains a numerical rating or feedback score between 1 and 10.
        
        IMPORTANT: Only return a number if the message is EXPLICITLY providing feedback on a previous intervention.
        Do NOT interpret expressions of stress, anxiety, or other emotions as feedback.
        
        If the message contains a numerical rating (1-10), extract only the number.
        If the message is EXPLICITLY providing positive feedback without a number (e.g., "that was helpful", "I feel better"), return 8.
        If the message is EXPLICITLY providing negative feedback without a number (e.g., "that didn't help", "I still feel anxious"), return 3.
        If the message is NOT providing feedback on a previous intervention, return "None".
        
        User message: "{message}"
        
        Respond with ONLY the number (1-10) or "None", nothing else.
        """
        print("\n" + "="*80)
        print("FEEDBACK DETECTION PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        # If we have a custom LLM client, use it
        if self.llm_client:
            raw_response = self.llm_client.generate(prompt)
        
        # Otherwise, use OpenAI
        else:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that detects feedback scores in user messages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0  # Use low temperature for more deterministic responses
            )
            raw_response = response.choices[0].message.content.strip()
            
        # Clean up the response
        response = raw_response.strip().lower()
        
        # Check if the response is "None" or empty
        if response in ["none", ""]:
            return None
            
        # Try to extract a number from the response
        try:
            score = int(response)
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass
            
        return None
        
    def _process_feedback(self, feedback_score: int, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback on a stress relief intervention.
        
        Args:
            feedback_score: The user's feedback score (1-10).
            user_id: The ID of the user.
            state: The current conversation state.
            
        Returns:
            A dictionary containing the response text and metadata.
        """
        # Safely extract data from state
        category = state.get('category', 'general stress')
        intervention = state.get('intervention', {})
        intervention_type = state.get('intervention_type') or intervention.get('type', 'unknown')
        intervention_name = intervention.get('name', intervention_type)
        feedback_message = state.get('feedback_message', '')
        
        # Generate a summary of the interaction
        summary = self._generate_interaction_summary(category, intervention, feedback_score, feedback_message)
        
        # Update the user's profile with the feedback score and summary
        try:
            # Try with intervention_name parameter
            self.profile_manager.update_score(
                user_id,
                category,
                intervention_type,
                feedback_score,
                summary,
                feedback_message,
                intervention_name
            )
        except TypeError:
            # Fall back to version without intervention_name
            try:
                self.profile_manager.update_score(
                    user_id,
                    category,
                    intervention_type,
                    feedback_score,
                    summary,
                    feedback_message
                )
            except TypeError:
                # Fall back to version without feedback_message
                self.profile_manager.update_score(
                    user_id,
                    category,
                    intervention_type,
                    feedback_score,
                    summary
                )
        
        # Determine if we should offer another suggestion
        if feedback_score < 4:
            # Negative feedback - offer a different intervention
            response_text = f"I'm sorry that didn't help. It takes time to find what works best for you. I will try to suggest something else that might be more effective in the future. "
        elif feedback_score >= 7:
            # Positive feedback - ask if they want another suggestion
            response_text = f"I'm glad that helped! "
        else:
            # Neutral feedback - acknowledge and ask if they want to try something else
            response_text = f"Thank you for your feedback! "
        
        # Update state to indicate feedback has been processed
        state['feedback_processed'] = True
        state['awaiting_feedback'] = False
        
        # Return a dictionary with the response text and metadata
        return {
            "text": response_text,
            "meta": {
                "category": category,
                "intervention_type": intervention_type,
                "intervention_name": intervention_name,
                "feedback_score": feedback_score,
                "feedback_message": feedback_message,
                "summary": summary
            }
        }

    def _generate_interaction_summary(self, category: str, intervention: Dict[str, Any], feedback_score: int, feedback_message: str) -> str:
        """Generate a summary of the interaction using LLM.
        
        Args:
            category: The stress category.
            intervention: The intervention details.
            feedback_score: The feedback score (1-10).
            feedback_message: The user's feedback message.
            
        Returns:
            A concise summary of the interaction.
        """
        # Safely extract intervention details
        intervention_name = intervention.get('name', 'unknown intervention')
        intervention_type = intervention.get('type', 'general')
        
        # Create the feedback part of the prompt only if there's actual feedback
        feedback_prompt_section = ""
        if feedback_message and feedback_message.strip():
            feedback_prompt_section = f"User's Feedback: \"{feedback_message}\"\n"
        
        # Create a prompt for the LLM
        prompt = f"""
        Create a very brief, action-oriented summary of a stress relief interaction with the following details:
        
        Category: {category}
        Intervention: {intervention_name}
        Intervention Type: {intervention_type}
        Feedback Score: {feedback_score}/10
        {feedback_prompt_section}
        The summary should be in the format: "[Initial state], [Intervention applied], [User's reaction]"
        For example: "User experiencing work anxiety, tried deep breathing exercise, reported feeling calmer afterward."
        
        Keep it extremely concise (15-30 words maximum).
        Make sure to include the user's reaction based on their feedback score{" and message" if feedback_prompt_section else ""}.
        """
        print("\n" + "="*80)
        print("INTERACTION SUMMARY PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        # Use the LLM to generate the summary
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise summaries of stress relief interactions, focusing on the user's condition, the intervention applied, and their reaction."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Create a basic summary if LLM fails
            sentiment = "positively" if feedback_score >= 7 else "negatively" if feedback_score < 4 else "neutrally"
            return f"User with {category} tried {intervention_name} and responded {sentiment} (score: {feedback_score}/10)."

    def _generate_feedback_response(self, feedback_score: int, user_id: str, state: Dict[str, Any]) -> str:
        """Generate a response based on the user's feedback.
        
        Args:
            feedback_score: The user's feedback score (1-10).
            user_id: The user's ID.
            state: The current state of the conversation.
            
        Returns:
            A response message based on the feedback.
        """
        # Get the current category and intervention
        category = state.get('category', 'general stress')
        intervention = state.get('intervention', {})
        
        # Generate a response based on the feedback score
        if feedback_score >= 7:
            response = f"I'm glad to hear that {intervention.get('name', 'the intervention')} was helpful! "
        else:
            response = f"I understand that {intervention.get('name', 'the intervention')} wasn't very helpful. "
            response += "Sometimes it takes a few tries to find what works best for you. "
            response += "Feel free to reach out again when you're ready to try something else."
        
        return response

    def _select_intervention(self, category: str, user_id: str) -> Tuple[str, Dict[str, Any]]:
        """Select the most appropriate intervention based on user history and preferences.
        
        Args:
            category: The stress category.
            user_id: The user's ID.
            
        Returns:
            A tuple of (intervention_type, intervention_details).
        """
        # Get user's intervention scores
        profile = self.profile_manager.get_profile(user_id)
        scores = profile.get("scores", {})
        
        # Get feedback analysis
        feedback_data = self.profile_manager.get_feedback_analysis(user_id)
        
        # Create a flat list of all interventions with their types for easier indexing
        all_interventions = []
        for intervention_type, interventions in self.intervention_db.interventions.items():
            if isinstance(interventions, list):
                for intervention in interventions:
                    intervention_with_type = intervention.copy()
                    intervention_with_type['type'] = intervention_type
                    all_interventions.append(intervention_with_type)
            else:
                intervention_with_type = interventions.copy() if isinstance(interventions, dict) else {'name': interventions}
                intervention_with_type['type'] = intervention_type
                all_interventions.append(intervention_with_type)
        
        # Format the interventions list without checkmarks
        interventions_text = ""
        for i, intervention in enumerate(all_interventions):
            intervention_type = intervention.get('type', 'unknown')
            interventions_text += f"{i}. {intervention.get('name', 'Unknown')} ({intervention_type})\n"
        
        # Format the scores text
        scores_text = ""
        for intervention_type, score in scores.items():
            scores_text += f"{intervention_type}: {score}\n"
        
        # Get recent interaction history
        history = profile.get("interaction_history", [])
        history_text = ""
        for interaction in history[-3:]:  # Get the last 3 interactions
            history_text += f"- {interaction.get('intervention_name', 'Unknown')} ({interaction.get('category', 'Unknown')}): {interaction.get('feedback_score', 0)}/10\n"
            if interaction.get('feedback_message'):
                history_text += f"  User's feedback: \"{interaction.get('feedback_message')}\"\n"
        
        # Add user preferences based on feedback
        user_preferences = ""
        if feedback_data.get("favorite_categories"):
            user_preferences += f"User's favorite categories: {', '.join(feedback_data['favorite_categories'])}\n"
        
        if feedback_data.get("recent_feedback"):
            # Extract insights from recent feedback
            for feedback in feedback_data["recent_feedback"][:2]:  # Just use the 2 most recent ones
                if feedback.get("feedback_score", 0) >= 7:
                    user_preferences += f"User responded positively to {feedback.get('intervention_name', 'Unknown')}\n"
                elif feedback.get("feedback_score", 0) <= 3:
                    user_preferences += f"User responded negatively to {feedback.get('intervention_name', 'Unknown')}\n"
        
        # Set up the prompt for the LLM
        prompt = f"""
        Select the most appropriate stress relief intervention based on the user's history and preferences.
        
        Current stress category: {category}
        
        Available interventions:
        {interventions_text}
        
        User's intervention scores:
        {scores_text}
        
        Recent interaction history:
        {history_text}
        
        User preferences and patterns:
        {user_preferences}
        
        Choose ONE intervention from the available list that would be most helpful for this user right now.
        Consider their preferences (higher scores), but also provide variety if they've used similar interventions repeatedly.
        Take into account their feedback messages to understand what works well for them.
        
        Return ONLY the index number of the chosen intervention, nothing else.
        """
        print("\n" + "="*80)
        print("INTERVENTION SELECTION PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        # If we have a custom LLM client, use it
        if self.llm_client:
            raw_response = self.llm_client.generate(prompt)
        
        # Otherwise, use OpenAI
        else:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that selects the best stress relief intervention based on user history and feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0  # Use low temperature for more deterministic responses
            )
            raw_response = response.choices[0].message.content.strip()
            
        # Parse the response to get the intervention index
        try:
            intervention_index = int(raw_response.strip())
            if 0 <= intervention_index < len(all_interventions):
                selected_intervention = all_interventions[intervention_index]
                intervention_type = selected_intervention.get('type')
                return intervention_type, selected_intervention
            else:
                raise IndexError(f"Index {intervention_index} out of range")
                
        except (ValueError, IndexError) as e:
            print(f"Error selecting intervention: {e}")
            # Select a random intervention
            selected = random.choice(all_interventions)
            return selected.get('type'), selected
    
    def _generate_response(self, message: str, category: str, 
                           intervention: Dict[str, Any], user_id: str) -> str:
        """Generate a personalized response using LLM.
        
        Args:
            message: The user's message.
            category: The detected stress category.
            intervention: The selected intervention.
            user_id: The user's ID.
            
        Returns:
            A personalized response message.
        """
        try:
            # Get user profile and interaction history
            profile = self.profile_manager.get_profile(user_id)
            history = profile.get("interaction_history", [])
            
            # Format the history for the prompt
            history_text = ""
            for interaction in history[-3:]:  # Get the last 3 interactions
                history_text += f"- {interaction.get('intervention_name', 'Unknown')} ({interaction.get('category', 'Unknown')}): {interaction.get('feedback_score', 0)}/10\n"
                if interaction.get('feedback_message'):
                    history_text += f"  Feedback: \"{interaction.get('feedback_message')}\"\n"
            
            # Set up the system prompt
            system_prompt = """You are a supportive stress relief assistant. Your responses should be:
            - Empathetic and understanding
            - Brief and focused
            - Action-oriented
            - Professional but warm
            - Based on evidence-based stress relief techniques
            """
            
            # Safely get intervention details
            intervention_name = intervention.get('name', 'stress relief technique')
            intervention_description = intervention.get('description', 'a helpful stress relief technique')
            intervention_link = intervention.get('link', '')
            
            user_prompt = f"""
            The user message is: "{message}"
            
            The user seems to be experiencing {category}.
            
            Recommend this specific practice: {intervention_name} - {intervention_description}
            
            End your message with a link to the practice: {intervention_link}
            
            User's recent intervention history:
            {history_text}
            
            Keep your response short (2-3 sentences). Be empathetic but avoid lengthy explanations.
            End by asking: "On a scale of 1 to 10, how helpful was this suggestion? Please also share how it made you feel or any other thoughts you have about it."
            """
            print("\n" + "="*80)
            print("RESPONSE GENERATION PROMPT:")
            print("="*80)
            print("System Prompt:")
            print(system_prompt)
            print("\nUser Prompt:")
            print(user_prompt)
            print("="*80 + "\n")
            
            # If we have a custom LLM client, use it
            if self.llm_client:
                response = self.llm_client.generate(user_prompt)
            
            # Otherwise, use OpenAI
            else:
                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=150,  # Limit token count for shorter responses
                    temperature=0.7  # Slightly higher temperature for more natural responses
                )
                response = completion.choices[0].message.content.strip()
                
            # Make sure the response includes the link if one was provided
            if intervention_link and intervention_link not in response:
                response += f"\n\n{intervention_link}"
                
            # Make sure the response asks for detailed feedback
            feedback_prompt = "On a scale of 1 to 10, how helpful was this suggestion? Please also share how it made you feel or any other thoughts you have about it."
            if feedback_prompt not in response and "scale of 1 to 10" not in response.lower():
                response += f"\n\n{feedback_prompt}"
                
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            return f"I recommend trying {intervention.get('name', 'a stress relief technique')} for your {category}. It may help you feel better. On a scale of 1 to 10, how helpful was this suggestion? Please also share how it made you feel."

    def process_user_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """Process a user message following the defined workflow sequence.
        
        Args:
            message: The user's message.
            user_id: The user's ID.
            
        Returns:
            A dictionary containing the response and metadata.
        """
        print("\n" + "="*80)
        print("PROCESSING USER MESSAGE")
        print("="*80)
        
        # Step 1: Load User Profile
        print("\n1. LOADING USER PROFILE")
        user_profile = self.profile_manager.get_profile(user_id)
        print(f"User profile loaded for user ID: {user_id}")
        
        # Step 2: Category Detection (LLM)
        print("\n2. CATEGORY DETECTION (LLM)")
        result = self.category_detector.detect_category(message)
        
        # Handle the case where category detector returns (category, confidence) or just category
        if isinstance(result, tuple):
            category, confidence = result
        else:
            category = result
            confidence = 1.0
            
        print(f"Detected category: {category} (confidence: {confidence:.2f})")
        
        # Step 3: Intervention Selection (LLM + user scores)
        print("\n3. INTERVENTION SELECTION (LLM + user scores)")
        intervention_type, intervention = self._select_intervention(category, user_id)
        print(f"Selected intervention: {intervention['name']} ({intervention_type})")
        
        # Step 4: LLM Prompt Construction
        print("\n4. LLM PROMPT CONSTRUCTION")
        # Get recent interaction history for context
        history = user_profile.get("interaction_history", [])
        history_text = ""
        for interaction in history[-3:]:  # Get the last 3 interactions
            history_text += f"- {interaction.get('intervention_name', 'Unknown')} ({interaction.get('category', 'Unknown')}): {interaction.get('feedback_score', 0)}/10\n"
        
        # Set up the system prompt
        system_prompt = """You are a supportive stress relief assistant. Your responses should be:
        - Empathetic and understanding
        - Brief and focused
        - Action-oriented
        - Professional but warm
        - Based on evidence-based stress relief techniques
        """
        
        user_prompt = f"""
        The user message is: "{message}"
        
        The user seems to be experiencing {category}.
        
        Recommend this specific practice: {intervention['name']} - {intervention['description']}
        
        End your message with a link to the practice: {intervention['link']}
        
        User's recent intervention history:
        {history_text}
        
        Keep your response short (2-3 sentences). Be empathetic but avoid lengthy explanations.
        Always end by asking: "On a scale of 1 to 10, how helpful was this suggestion?"
        """
        print("Prompts constructed successfully")
        
        # Step 5: Return Intervention Suggestion
        print("\n5. RETURNING INTERVENTION SUGGESTION")
        response = self._generate_response(message, category, intervention, user_id)
        print("Response generated successfully")
        
        # Step 6: Ask for Feedback
        print("\n6. ASKING FOR FEEDBACK")
        # The feedback request is included in the response
        print("Feedback request included in response")
        
        # Step 7: Update User Profile
        print("\n7. UPDATING USER PROFILE")
        # Store the current state for feedback processing
        self.current_state[user_id] = {
            "timestamp": self.time.time(),
            "category": category,
            "intervention_type": intervention_type,
            "intervention": intervention,
            "awaiting_feedback": True,
            "feedback_processed": False
        }
        print("User profile updated with current interaction state")
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80 + "\n")
        
        # Return the response and metadata
        return {
            "text": response,
            "meta": {
                "category": category,
                "confidence": confidence,
                "intervention_type": intervention_type,
                "intervention_name": intervention['name'],
                "intervention_link": intervention.get('link', '')
            }
        }
        
    def process_feedback(self, message: str, user_id: str) -> Dict[str, Any]:
        """Process feedback from a user message.
        
        Args:
            message: The user's message containing feedback.
            user_id: The user's ID.
            
        Returns:
            A dictionary containing the response and metadata.
        """
        # Detect feedback score
        feedback_score = self._llm_detect_feedback(message)
        
        # If no feedback detected, return None
        if feedback_score is None:
            return None
            
        # Get the current state for this user
        if user_id not in self.current_state:
            print(f"No current state found for user {user_id}")
            return {
                "text": "I'm sorry, but I don't have any context for your feedback. Could you tell me more about what you're experiencing?",
                "meta": {
                    "error": "No state found",
                    "feedback_score": feedback_score
                }
            }
        
        # Store the user's feedback message
        self.current_state[user_id]["feedback_message"] = message
        
        # Process the feedback using the current state
        return self._process_feedback(feedback_score, user_id, self.current_state[user_id])

    def analyze_user_feedback(self, user_id: str) -> Dict[str, Any]:
        """Analyze a user's feedback and reactions to interventions.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            A dictionary containing feedback analysis.
        """
        # Get feedback analysis from the profile manager
        feedback_data = self.profile_manager.get_feedback_analysis(user_id)
        
        # Only proceed if we have feedback data
        if not feedback_data.get("recent_feedback"):
            return {
                "text": "Not enough feedback data available yet.",
                "meta": feedback_data
            }
        
        # Construct a prompt to analyze the feedback
        recent_feedback = feedback_data["recent_feedback"]
        feedback_text = ""
        for i, feedback in enumerate(recent_feedback):
            feedback_text += f"{i+1}. Category: {feedback['category']}\n"
            feedback_text += f"   Intervention: {feedback['intervention_name']}\n"
            feedback_text += f"   Score: {feedback['feedback_score']}/10\n"
            if feedback.get("feedback_message"):
                feedback_text += f"   Message: \"{feedback['feedback_message']}\"\n"
            if feedback.get("summary"):
                feedback_text += f"   Summary: {feedback['summary']}\n"
            feedback_text += "\n"
        
        prompt = f"""
        Analyze the following user feedback on stress relief interventions:
        
        {feedback_text}
        
        Based on this feedback history, please provide:
        1. What interventions seem most effective for this user?
        2. What patterns do you notice in their responses?
        3. What types of interventions should be recommended more often?
        4. What interventions should be avoided?
        
        Keep your analysis concise (100-150 words).
        """
        
        print("\n" + "="*80)
        print("FEEDBACK ANALYSIS PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        try:
            # Use the LLM to analyze the feedback
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes patterns in user feedback to improve stress relief recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            analysis = response.choices[0].message.content.strip()
            
            # Return the analysis along with the metadata
            return {
                "text": analysis,
                "meta": {
                    **feedback_data,
                    "analysis_provided": True
                }
            }
        except Exception as e:
            print(f"Error analyzing feedback: {e}")
            return {
                "text": "Unable to analyze feedback at this time.",
                "meta": {
                    **feedback_data,
                    "error": str(e)
                }
            }
            
    def get_user_feedback_history(self, user_id: str, limit: int = 5) -> Dict[str, Any]:
        """Get a user's feedback history.
        
        Args:
            user_id: The user's ID.
            limit: The maximum number of interactions to return.
            
        Returns:
            A dictionary containing the feedback history.
        """
        # Get recent interactions from the profile manager
        recent_interactions = self.profile_manager.get_recent_interactions(user_id, limit)
        
        # Format the history for display
        formatted_history = []
        for interaction in recent_interactions:
            formatted_interaction = {
                "date": interaction.get("timestamp", "").split("T")[0],
                "category": interaction.get("category", "unknown"),
                "intervention": interaction.get("intervention_name", interaction.get("intervention_type", "unknown")),
                "score": interaction.get("feedback_score", 0),
                "feedback": interaction.get("feedback_message", "No feedback provided")
            }
            
            # Add summary if available
            if "summary" in interaction:
                formatted_interaction["summary"] = interaction["summary"]
                
            formatted_history.append(formatted_interaction)
            
        return {
            "text": f"Retrieved {len(formatted_history)} recent feedback entries.",
            "meta": {
                "history": formatted_history,
                "user_id": user_id
            }
        }

    def handle_feedback(self, message: str, user_id: str) -> Dict[str, Any]:
        """Explicitly process a message as feedback for the previous intervention.
        
        Args:
            message: The user's feedback message.
            user_id: The user's ID.
            
        Returns:
            A dictionary containing the response and metadata, or None if no feedback detected.
        """
        # Only process if there is state to process feedback against
        if user_id not in self.current_state:
            return {
                "text": "No previous intervention found to provide feedback on.",
                "meta": {
                    "error": "No state found"
                }
            }
            
        # Detect and process feedback
        feedback_score = self._llm_detect_feedback(message)
        if feedback_score is not None:
            # Store the feedback message
            self.current_state[user_id]["feedback_message"] = message
            # Process the feedback
            return self._process_feedback(feedback_score, user_id, self.current_state[user_id])
        else:
            return {
                "text": "I couldn't detect specific feedback about the previous suggestion. Could you please rate it from 1-10?",
                "meta": {
                    "error": "No feedback detected"
                }
            }

