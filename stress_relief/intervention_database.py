from typing import Dict, List, Any

class InterventionDatabase:
    """Database of stress relief interventions."""

    def __init__(self):
        """Initialize the intervention database."""
        self.interventions = {
            # 1. Relaxation (Breath Awareness / Centering)
            "relaxation": [
                {
                    "name": "60-second breathing",
                    "link": "https://www.youtube.com/watch?v=Dx112W4i5I0",
                    "description": "A brief guided breathing exercise designed to reduce stress and promote relaxation."
                },
                {
                    "name": "Box Breathing",
                    "link": "https://www.youtube.com/watch?v=tEmt1Znux58",
                    "description": "A simple relaxation technique involving four-second intervals of inhaling, holding, exhaling, and holding the breath to calm feelings of stress or anxiety."
                },
                {
                    "name": "Muscle Relaxation",
                    "link": "https://youtu.be/ClqPtWzozXs?t=69",
                    "description": "A progressive muscle relaxation exercise that helps release physical tension and promote a sense of calm."
                }
            ],

            # 2. Reappraisal (Cognitive Restructuring)
            "reappraisal": [
                {
                    "name": "Script from Wiley SMI",
                    "link": "https://onlinelibrary.wiley.com/doi/full/10.1002/smi.2759",
                    "description": "A guided cognitive reappraisal script to help reinterpret negative experiences objectively, fostering emotional resilience."
                }
            ],

            # 3. Positive Experiences & Mood Induction
            "positive_experiences": [
                {
                    "name": "Funny Animal Video",
                    "link": "https://www.youtube.com/watch?v=rzr6Zv6Mfbs",
                    "description": "A short, uplifting video featuring humorous animal antics to boost mood and alleviate stress."
                },
                {
                    "name": "Guided Imagery",
                    "link": "https://www.youtube.com/watch?v=QtE00VP4W3Y",
                    "description": "A quick guided meditation focusing on visualization to help reset and refocus the mind."
                }
            ],

            # 4. Gratitude
            "gratitude": [
                {
                    "name": "2-min Gratitude Meditation",
                    "link": "https://www.youtube.com/watch?v=OCorElLKFQE",
                    "description": "A brief meditation encouraging reflection on aspects of life you're grateful for, aimed at enhancing mood."
                }
            ],

            # 5. Resource Buffers (Self-Affirmation & Self-Efficacy)
            "resource_buffers": [
                {
                    "name": "Self-Affirmation Video",
                    "link": "https://www.youtube.com/watch?v=qANaxInPFh0",
                    "description": "A short video promoting self-kindness and positive self-talk to reinforce self-worth."
                },
                {
                    "name": "Best Possible Self",
                    "link": "https://www.youtube.com/watch?v=G_jEsnDEIa0",
                    "description": "An exercise guiding you to envision your ideal future self, fostering optimism and motivation."
                }
            ]
        }

        # Mapping from stress categories to intervention types
        self.category_to_intervention = {
            "anxiety": ["relaxation", "resource_buffers"],
            "sadness": ["gratitude", "positive_experiences"],
            "anger": ["reappraisal", "relaxation"],
            "overwhelm": ["relaxation", "resource_buffers"],
            "general": ["positive_experiences", "gratitude"]
}

    def get_interventions_for_category(self, category: str) -> List[str]:
        """Get intervention types suitable for a specific stress category."""
        return self.category_to_intervention.get(category, ["relaxation", "positive_experiences"])

    def get_intervention(self, intervention_type: str, index: int = 0) -> Dict[str, Any]:
        """Get a specific intervention."""
        interventions = self.interventions.get(intervention_type, [])
        if not interventions:
            # Return a default intervention if the type doesn't exist
            return {
                "name": "Breathing Exercise",
                "link": "https://www.youtube.com/watch?v=Dx112W4i5I0",
                "description": "A brief guided breathing exercise designed to reduce stress and promote relaxation."
            }

        # Wrap around if index is out of bounds
        index = index % len(interventions)
        return interventions[index]
