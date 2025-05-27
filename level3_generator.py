"""
Level-3 Question Generator for code data.
Main orchestrator that combines all components to generate high-quality programming questions.
"""

import re
from typing import List, Dict, Any, Optional
from .data_structures import CodeConcept, ExtractedContent, GeneratedQuestion, LLMInterface
from .knowledge_graph import KnowledgeGraph
from .prompts import PromptTemplates


class ContentParser:
    """Parses LLM responses to extract structured content."""
    
    @staticmethod
    def parse_concept_extraction_response(response: str, source_document: str) -> ExtractedContent:
        """Parse the LLM response from concept extraction prompt."""
        try:
            # Extract educational level
            level_match = re.search(r'<level>(.*?)</level>', response, re.DOTALL)
            educational_level = level_match.group(1).strip() if level_match else "intermediate"
            
            # Extract subject area
            subject_match = re.search(r'<subject>(.*?)</subject>', response, re.DOTALL)
            subject_area = subject_match.group(1).strip() if subject_match else "Programming"
            
            # Extract topics
            topic_match = re.search(r'<topic>(.*?)</topic>', response, re.DOTALL)
            topics = []
            if topic_match:
                topic_text = topic_match.group(1)
                # Parse numbered list
                topic_lines = [line.strip() for line in topic_text.split('\n') if line.strip()]
                for line in topic_lines:
                    if re.match(r'^\d+\.', line.strip()):
                        topic = re.sub(r'^\d+\.\s*', '', line.strip())
                        if topic:
                            topics.append(topic)
            
            # Extract key concepts
            concept_match = re.search(r'<key_concept>(.*?)</key_concept>', response, re.DOTALL)
            key_concepts = {}
            if concept_match:
                concept_text = concept_match.group(1)
                current_topic = None
                
                for line in concept_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Topic line (e.g., "1. topic 1:")
                    topic_match = re.match(r'^\d+\.\s*(.+?):\s*$', line)
                    if topic_match:
                        current_topic = topic_match.group(1).strip()
                        key_concepts[current_topic] = []
                        continue
                    
                    # Concept line (e.g., "1.1. key concept")
                    concept_match = re.match(r'^\d+\.\d+\.\s*(.+)$', line)
                    if concept_match and current_topic:
                        concept = concept_match.group(1).strip()
                        if concept:
                            key_concepts[current_topic].append(concept)
            
            return ExtractedContent(
                educational_level=educational_level,
                subject_area=subject_area,
                topics=topics,
                key_concepts=key_concepts,
                source_document=source_document
            )
            
        except Exception as e:
            print(f"Error parsing concept extraction response: {e}")
            # Return default structure
            return ExtractedContent(
                educational_level="intermediate",
                subject_area="Programming",
                topics=["General Programming"],
                key_concepts={"General Programming": ["Programming Concepts"]},
                source_document=source_document
            )
    
    @staticmethod
    def parse_question_generation_response(response: str, 
                                         concepts_used: List[CodeConcept],
                                         source_documents: List[str]) -> List[GeneratedQuestion]:
        """Parse the LLM response from question generation prompt."""
        questions = []
        
        try:
            # Find all question blocks
            question_matches = re.findall(r'<Q\d+>(.*?)</Q\d+>', response, re.DOTALL)
            
            for match in question_matches:
                # Extract selected concepts
                concept_match = re.search(r'Selected Concepts:\s*\[(.*?)\]', match, re.DOTALL)
                selected_concepts = []
                if concept_match:
                    concept_text = concept_match.group(1)
                    selected_concepts = [c.strip() for c in concept_text.split(',') if c.strip()]
                
                # Extract question
                question_match = re.search(r'Question:\s*(.*?)(?=\n\n|\Z)', match, re.DOTALL)
                question_text = question_match.group(1).strip() if question_match else ""
                
                if question_text and selected_concepts:
                    questions.append(GeneratedQuestion(
                        question=question_text,
                        selected_concepts=selected_concepts,
                        concepts_used=[c.name for c in concepts_used],
                        concept_categories=[c.category for c in concepts_used],
                        complexity_levels=[c.complexity_level for c in concepts_used],
                        source_documents=source_documents,
                        educational_level=concepts_used[0].educational_level if concepts_used else "intermediate",
                        subject_area=concepts_used[0].source_document if concepts_used else "Programming"
                    ))
        
        except Exception as e:
            print(f"Error parsing question generation response: {e}")
        
        return questions


class Level3QuestionGenerator:
    """
    Level-3 Question Generator that creates high-quality programming questions
    by combining concepts from multiple documents and leveraging knowledge graphs.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.knowledge_graph = KnowledgeGraph()
        self.processed_documents: List[str] = []
        self.document_contents: Dict[str, str] = {}  # Store original document content
        self.parser = ContentParser()
        self.prompt_templates = PromptTemplates()
    
    def process_documents(self, documents: List[str], document_ids: Optional[List[str]] = None) -> None:
        """Process multiple reference documents to build the knowledge graph."""
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        print(f"Processing {len(documents)} documents...")
        
        for i, (document, doc_id) in enumerate(zip(documents, document_ids)):
            print(f"Processing document {i+1}/{len(documents)}: {doc_id}")
            
            try:
                # Store document content for later use in question generation
                self.document_contents[doc_id] = document
                
                # Extract concepts using LLM
                prompt = self.prompt_templates.format_concept_extraction_prompt(document)
                extracted_content = self.llm.extract_concepts(document, prompt)
                
                # Convert extracted content to CodeConcept objects
                for topic, concepts in extracted_content.key_concepts.items():
                    for concept_name in concepts:
                        # Find related concepts within the same extraction
                        related_concepts = []
                        for other_topic, other_concepts in extracted_content.key_concepts.items():
                            if other_topic != topic:
                                related_concepts.extend(other_concepts[:3])  # Limit to avoid too many connections
                        
                        concept = CodeConcept(
                            name=concept_name,
                            category=self._categorize_concept(concept_name, topic),
                            description=f"