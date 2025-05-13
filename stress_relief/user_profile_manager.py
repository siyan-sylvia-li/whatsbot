import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class UserProfileManager:
    """Manages user profiles for the stress relief module."""
    
    def __init__(self, profile_path: str = "user_profiles.json"):
        """Initialize the UserProfileManager."""
        self.profile_path = profile_path
        self.profiles = self._load_profiles()
        
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load user profiles from the JSON file."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_profiles(self) -> None:
        """Save user profiles to the JSON file."""
        with open(self.profile_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Get a user's profile, creating it if it doesn't exist."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "scores": {
                    "relaxation": 0.0,
                    "reappraisal": 0.0,
                    "positive_experiences": 0.0,
                    "gratitude": 0.0,
                    "resource_buffers": 0.0
                },
                "favorites": [],
                "interaction_history": []
            }
            self._save_profiles()
        return self.profiles[user_id]
    
    def update_score(self, user_id: str, category: str, intervention_type: str, 
                     feedback_score: int, summary: str = None, feedback_message: str = None,
                     intervention_name: str = None) -> None:
        """Update a user's score for a particular intervention category and store feedback.
        
        Args:
            user_id: The user's ID.
            category: The stress category.
            intervention_type: The type of intervention.
            feedback_score: The feedback score (1-10).
            summary: Optional summary of the interaction.
            feedback_message: Optional message containing the user's feedback.
            intervention_name: Optional specific name of the intervention.
        """
        profile = self.get_profile(user_id)
        
        # Update the interaction history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "intervention_type": intervention_type,
            "intervention_name": intervention_name if intervention_name else intervention_type,
            "feedback_score": feedback_score
        }
        
        # Add summary if provided
        if summary:
            interaction["summary"] = summary
            
        # Add feedback message if provided
        if feedback_message:
            interaction["feedback_message"] = feedback_message
            
        # Store in interaction history
        profile["interaction_history"].append(interaction)
        
        # Update the rolling average for the intervention type
        scores = profile["scores"]
        
        # Count how many times this intervention type has been used
        count = sum(1 for interaction in profile["interaction_history"] 
                    if interaction["intervention_type"] == intervention_type)
        
        # Calculate the new average
        if intervention_type in scores:
            old_avg = scores[intervention_type]
            scores[intervention_type] = ((old_avg * (count - 1)) + feedback_score) / count
        else:
            scores[intervention_type] = feedback_score
        
        # Update favorites (categories consistently rated ≥ 7.5)
        profile["favorites"] = [
            intervention_type for intervention_type, score in scores.items()
            if score >= 7.5
        ]
        
        # Save the updated profiles
        self._save_profiles()
    
    def get_favorite_intervention(self, user_id: str) -> Optional[str]:
        """Get the user's favorite intervention type."""
        profile = self.get_profile(user_id)
        favorites = profile["favorites"]
        
        if not favorites:
            return None
        
        # Return the highest-rated favorite
        scores = profile["scores"]
        return max(favorites, key=lambda x: scores.get(x, 0))
    
    def get_intervention_score(self, user_id: str, intervention_type: str) -> float:
        """Get a user's score for a particular intervention type."""
        profile = self.get_profile(user_id)
        return profile["scores"].get(intervention_type, 0.0)
    
    def get_recent_interactions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent interactions for a user.
        
        Args:
            user_id: The user's ID.
            limit: The maximum number of interactions to return.
            
        Returns:
            A list of the most recent interactions.
        """
        profile = self.get_profile(user_id)
        history = profile.get("interaction_history", [])
        
        # Sort by timestamp (most recent first)
        sorted_history = sorted(
            history, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        return sorted_history[:limit]
        
    def get_feedback_analysis(self, user_id: str) -> Dict[str, Any]:
        """Analyze a user's feedback patterns.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            A dictionary containing feedback analysis.
        """
        profile = self.get_profile(user_id)
        history = profile.get("interaction_history", [])
        
        if not history:
            return {
                "total_interactions": 0,
                "average_score": 0,
                "favorite_categories": [],
                "recent_feedback": []
            }
        
        # Calculate basic metrics
        scores = [entry.get("feedback_score", 0) for entry in history]
        average_score = sum(scores) / len(scores) if scores else 0
        
        # Get category preferences
        category_counts = {}
        for entry in history:
            category = entry.get("category")
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        favorite_categories = sorted(category_counts.keys(), 
                                    key=lambda x: category_counts[x],
                                    reverse=True)
        
        # Get recent feedback with reactions
        recent_feedback = []
        for entry in sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:5]:
            if "feedback_message" in entry or "summary" in entry:
                recent_feedback.append({
                    "timestamp": entry.get("timestamp", ""),
                    "category": entry.get("category", "unknown"),
                    "intervention_type": entry.get("intervention_type", "unknown"),
                    "intervention_name": entry.get("intervention_name", entry.get("intervention_type", "unknown")),
                    "feedback_score": entry.get("feedback_score", 0),
                    "feedback_message": entry.get("feedback_message", ""),
                    "summary": entry.get("summary", "")
                })
        
        return {
            "total_interactions": len(history),
            "average_score": average_score,
            "favorite_categories": favorite_categories[:3],  # Top 3 categories
            "recent_feedback": recent_feedback
        } 