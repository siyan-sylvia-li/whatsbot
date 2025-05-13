import os
from openai import OpenAI
from .stress_relief_module import StressReliefModule

my_key = ""

def test_stress_relief():
    # Create a temporary profile path
    test_profile_path = "test_profiles.json"
    
    # Initialize the module with the API key directly
    bot = StressReliefModule(
        profile_path=test_profile_path,
        openai_api_key=my_key,  # Pass the API key directly
        favorite_threshold=7.0  # Use a custom threshold
    )
    
    # Test message processing
    user_message = "I've been feeling completely overwhelmed lately with my job, family responsibilities, and studying for exams. There are just too many things happening at once and I can't seem to keep up with everything. My mind is racing constantly and I'm having trouble sleeping."
    user_id = "test_user_2"
    
    # Process message (always treated as a new message)
    response = bot.process_message(user_message, user_id)
    
    # Display results
    print(f"User message: {user_message}")
    print(f"Response: {response['text']}")
    print(f"Category: {response['meta']['category']}")
    print(f"Confidence: {response['meta'].get('confidence', 'N/A')}")
    print(f"Intervention: {response['meta']['intervention_name']}")
    
    # Test feedback (explicitly using the feedback handler)
    feedback_message = "That was helpful, 8 out of 10. I'm feeling better now."
    feedback_response = bot.handle_feedback(feedback_message, user_id)
    
    print("\nFeedback processing:")
    print(f"Feedback message: {feedback_message}")
    print(f"Response: {feedback_response['text']}")

    # Clean up
    # if os.path.exists(test_profile_path):
    #     os.remove(test_profile_path)

if __name__ == "__main__":
    test_stress_relief() 