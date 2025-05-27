"""
Prompt templates for the Level-3 Code Question Generator.
Adapted from math prompts to focus on programming and software development.
"""


class PromptTemplates:
    """Collection of prompt templates for code data processing."""
    
    @staticmethod
    def get_concept_extraction_prompt() -> str:
        """
        Prompt for extracting programming concepts from technical documentation.
        Adapted from the original math concept extraction prompt.
        """
        return """prompt1"""

    @staticmethod
    def get_question_generation_prompt() -> str:
        """
        Prompt for generating programming questions from concept combinations.
        Adapted from the original math question generation prompt.
        """
        return """prompt2"""

    @staticmethod
    def format_concept_extraction_prompt(document_text: str) -> str:
        """Format the concept extraction prompt with document text."""
        return PromptTemplates.get_concept_extraction_prompt().replace("{{ text }}", document_text)
    
    @staticmethod
    def format_question_generation_prompt(reference_text: str, concept_list: str) -> str:
        """Format the question generation prompt with reference materials and concepts."""
        prompt = PromptTemplates.get_question_generation_prompt()
        prompt = prompt.replace("{{ text }}", reference_text)
        prompt = prompt.replace("{{ concept }}", concept_list)
        return prompt