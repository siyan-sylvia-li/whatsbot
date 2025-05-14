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
                },
                {
                    "name": "Breathing Exercise Reading",
                    "link": "https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/breathing-exercises-for-stress/",
                    "description": "A short passage describing breathing exercise techniques for stress."
                },
                {
                    "name": "Three Breathing Exercises",
                    "link": "https://www.bhf.org.uk/informationsupport/heart-matters-magazine/wellbeing/breathing-exercises",
                    "description": "An article describing three different breathing techniques for stress, including box breathing, the 4-7-8 breathing technique, and alternate nostril breathing."
                },
                {
                    "name": "Relaxation Exercises",
                    "link": "https://www.nhsinform.scot/healthy-living/mental-wellbeing/stress/breathing-and-relaxation-exercises/",
                    "description": "Various breathing and relaxation techniques for managing stress."
                },
                {
                    "name": "De-stressing Meditation",
                    "link": "https://www.youtube.com/watch?v=sG7DBA-mgFY",
                    "description": "A ten-minute meditation to reframe stress."
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
                    "name": "Funny Animal Videos",
                    "link": "https://www.youtube.com/results?search_query=funny+animal+videos",
                    "description": "A list of videos featuring humorous animal antics to boost mood and alleviate stress."
                },
                {
                    "name": "Guided Imagery",
                    "link": "https://www.youtube.com/watch?v=QtE00VP4W3Y",
                    "description": "A quick guided meditation focusing on visualization to help reset and refocus the mind."
                },
                {
                    "name": "Guided Imagery & Visualization: Anchoring",
                    "link": "https://students.dartmouth.edu/wellness-center/sites/students_wellness_center.prod/files/anchoring.mp3",
                    "description": "Anchoring is a hypnotic technique that helps you connect to times in your past when you felt truly calm and confident. You can use it right now to give yourself a feeling of strength when facing sad days and difficult challenges."
                },
                {
                    "name": "Guided Imagery & Visualization: The Forest",
                    "link": "https://students.dartmouth.edu/wellness-center/sites/students_wellness_center.prod/files/imagery_the_forest.mp3",
                    "description": "Let yourself be guided on a peaceful walk through a beautiful, lush forest near a trickling stream."
                },
                {
                    "name": "Guided Imagery & Visualization: Nourishment From The Past",
                    "link": "https://students.dartmouth.edu/wellness-center/sites/students_wellness_center.prod/files/nourish_from_past.mp3",
                    "description": "This Five-Finger exercise was developed by Dr. David Cheek as a way to achieve deep relaxation and peace, while simultaneously affirming your human worth. All you have to do is imagine four scenes from your past—using visual, auditory, and kinesthetic (touch) images. It’s simple, it’s pleasurable, and it works."
                },
                {
                    "name": "Guided Imagery & Visualization: Special Place",
                    "link": "https://students.dartmouth.edu/wellness-center/sites/students_wellness_center.prod/files/special_place.mp3",
                    "description": "This exercise guides you to create a safe and peaceful place in your imagination, a place you can go any time you need to relax. You should go there often, whenever tension starts to build. Merely close your eyes and focus on the image of your special place."
                },
                {
                    "name": "Guided Imagery & Visualization: Grounding",
                    "link": "https://students.dartmouth.edu/wellness-center/sites/students_wellness_center.prod/files/grounding.mp3",
                    "description": "This exercise guides you to create your personal shield, a shied that protects, nurtures and calms you. You can use it right now to give yourself a feeling of peace and calm."
                }
            ],

            # 4. Gratitude
            "gratitude": [
                {
                    "name": "5-min Gratitude Meditation",
                    "link": "https://www.youtube.com/watch?v=OCorElLKFQE",
                    "description": "A brief meditation encouraging reflection on aspects of life you're grateful for, aimed at enhancing mood."
                },
                {
                    "name": "Headspace Gratitude Meditation",
                    "link": "https://www.headspace.com/meditation/gratitude",
                    "description": "Free gratitude meditations from Headspace."
                },
                {
                    "name": "Gratitutde Journaling Prompts",
                    "link": "https://dayoneapp.com/blog/gratitude-journaling-prompts/",
                    "description": "Some prompts for gratitude journaling to help you reflect on things you are thankful for."
                },
                {
                    "name": "5-Minute Gratitude Meditation",
                    "link": "https://www.youtube.com/watch?v=zyUy9w953L0",
                    "description": "This short & Original guided meditation (recorded by us) will leave you feeling full of gratitude.  You will be amazed how grateful you are, and how positive it leaves you feeling.  Enjoy!"
                },
                {
                    "name": "Gratitude Affirmations",
                    "link": "https://blog.gratefulness.me/gratitude-affirmations/",
                    "description": "Here are 100 gratitude affirmations to manifest more gratitude and joy in your life!"
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
                },
                {
                    "name": "Self Affirmations",
                    "link": "https://www.youtube.com/watch?v=yo1pJ_D-H3M",
                    "description": "Powerful positive affirmations for self love, self esteem, confidence & self worth."
                },
                {
                    "name": "Good Things Are Happening to Me | Morning Affirmations",
                    "link": "https://www.youtube.com/watch?v=PcY4DjQPEyQ",
                    "description": "This morning, use the law of attraction and remind yourself that good things are happening to you. These morning affirmations will boost your mood and attract positive thinking. Good things are happening to you and me!"
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
