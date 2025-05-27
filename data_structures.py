"""
Core data structures for the Level-3 Code Question Generator.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class CodeConcept:
    """Represents a programming concept extracted from documentation."""
    name: str
    category: str  # e.g., 'data_structure', 'algorithm', 'design_pattern', 'syntax', 'framework'
    description: str
    complexity_level: str  # 'beginner', 'intermediate', 'advanced', 'expert'
    related_concepts: List[str]
    code_examples: List[str]
    source_document: str
    educational_level: str  # 'beginner', 'intermediate', 'advanced', 'professional', 'competition'


@dataclass
class ExtractedContent:
    """Structure for content extracted from documents."""
    educational_level: str
    subject_area: str  # e.g., 'Data Structures', 'Algorithms', 'Web Development', etc.
    topics: List[str]
    key_concepts: Dict[str, List[str]]  # topic -> list of concepts
    source_document: str


@dataclass
class GeneratedQuestion:
    """Structure for generated questions."""
    question: str
    selected_concepts: List[str]
    concepts_used: List[str]
    concept_categories: List[str]
    complexity_levels: List[str]
    source_documents: List[str]
    educational_level: str
    subject_area: str
    generation_method: str = "level-3"


class LLMInterface(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def extract_concepts(self, document: str, prompt: str) -> ExtractedContent:
        """Extract concepts from a document using concept extraction prompt."""
        pass
    
    @abstractmethod
    def generate_questions(self, 
                          concepts: List[CodeConcept], 
                          reference_materials: List[str],
                          prompt: str) -> List[GeneratedQuestion]:
        """Generate questions from concept combinations using question generation prompt."""
        pass


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes."""
    
    def extract_concepts(self, document: str, prompt: str) -> ExtractedContent:
        """Mock concept extraction - returns sample programming concepts."""
        return ExtractedContent(
            educational_level="intermediate",
            subject_area="Data Structures and Algorithms",
            topics=["Binary Trees", "Graph Algorithms", "Dynamic Programming"],
            key_concepts={
                "Binary Trees": [
                    "Binary Search Tree", "Tree Traversal", "AVL Trees", 
                    "Heap Operations", "Tree Balancing"
                ],
                "Graph Algorithms": [
                    "Depth-First Search", "Breadth-First Search", "Dijkstra's Algorithm",
                    "Topological Sort", "Minimum Spanning Tree"
                ],
                "Dynamic Programming": [
                    "Memoization", "Tabulation", "Optimal Substructure",
                    "Overlapping Subproblems", "State Transition"
                ]
            },
            source_document="mock_doc"
        )
    
    def generate_questions(self, 
                          concepts: List[CodeConcept], 
                          reference_materials: List[str],
                          prompt: str) -> List[GeneratedQuestion]:
        """Mock question generation."""
        concept_names = [c.name for c in concepts]
        
        return [
            GeneratedQuestion(
                question=f"Design and implement a system that combines {' and '.join(concept_names[:2])} to solve a real-world problem. Provide time and space complexity analysis.",
                selected_concepts=concept_names[:2],
                concepts_used=concept_names,
                concept_categories=[c.category for c in concepts],
                complexity_levels=[c.complexity_level for c in concepts],
                source_documents=list(set(c.source_document for c in concepts)),
                educational_level=concepts[0].educational_level if concepts else "intermediate",
                subject_area="Programming"
            ),
            GeneratedQuestion(
                question=f"Compare and contrast {concept_names[0]} with {concept_names[1] if len(concept_names) > 1 else 'alternative approaches'}. When would you choose one over the other?",
                selected_concepts=concept_names[:2],
                concepts_used=concept_names,
                concept_categories=[c.category for c in concepts],
                complexity_levels=[c.complexity_level for c in concepts],
                source_documents=list(set(c.source_document for c in concepts)),
                educational_level=concepts[0].educational_level if concepts else "intermediate",
                subject_area="Programming"
            )
        ]