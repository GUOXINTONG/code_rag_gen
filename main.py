import random
import json
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from abc import ABC, abstractmethod


@dataclass
class CodeConcept:
    """Represents a programming concept extracted from documentation."""
    name: str
    category: str  # e.g., 'data_structure', 'algorithm', 'design_pattern', 'syntax'
    description: str
    complexity_level: str  # 'basic', 'intermediate', 'advanced'
    related_concepts: List[str]
    code_examples: List[str]
    source_document: str


class LLMInterface(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def extract_concepts(self, document: str, prompt: str) -> List[Dict[str, Any]]:
        """Extract concepts from a document using prompt1."""
        pass
    
    @abstractmethod
    def generate_questions(self, concept_combination: List[CodeConcept], prompt: str) -> List[str]:
        """Generate questions from concept combinations using prompt2."""
        pass


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes."""
    
    def extract_concepts(self, document: str, prompt: str) -> List[Dict[str, Any]]:
        # Mock concept extraction - in real implementation, this calls your LLM
        mock_concepts = [
            {
                "name": "Binary Search Tree",
                "category": "data_structure", 
                "description": "A tree data structure where left child < parent < right child",
                "complexity_level": "intermediate",
                "related_concepts": ["Tree Traversal", "Recursion"],
                "code_examples": ["class BST: def insert(self, val): ..."]
            },
            {
                "name": "Dynamic Programming",
                "category": "algorithm",
                "description": "Optimization technique using memoization",
                "complexity_level": "advanced", 
                "related_concepts": ["Memoization", "Recursion"],
                "code_examples": ["def fibonacci(n, memo={}): ..."]
            }
        ]
        return mock_concepts
    
    def generate_questions(self, concept_combination: List[CodeConcept], prompt: str) -> List[str]:
        # Mock question generation
        concepts_str = ", ".join([c.name for c in concept_combination])
        return [
            f"How would you implement {concepts_str} together in Python?",
            f"What are the trade-offs when combining {concepts_str}?",
            f"Design a system that uses {concepts_str} efficiently."
        ]


class KnowledgeGraph:
    """Constructs and manages a knowledge graph of programming concepts."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.concepts: Dict[str, CodeConcept] = {}
        self.concept_categories: Dict[str, Set[str]] = defaultdict(set)
    
    def add_concept(self, concept: CodeConcept) -> None:
        """Add a concept to the knowledge graph."""
        self.concepts[concept.name] = concept
        self.concept_categories[concept.category].add(concept.name)
        
        # Add node to graph
        self.graph.add_node(concept.name, 
                           category=concept.category,
                           complexity=concept.complexity_level,
                           description=concept.description)
        
        # Add edges to related concepts
        for related in concept.related_concepts:
            if related in self.concepts:
                self.graph.add_edge(concept.name, related)
    
    def get_connected_concepts(self, concept_name: str, max_distance: int = 2) -> Set[str]:
        """Get concepts connected to the given concept within max_distance."""
        if concept_name not in self.graph:
            return set()
        
        connected = set()
        for target in self.graph.nodes():
            try:
                distance = nx.shortest_path_length(self.graph, concept_name, target)
                if 0 < distance <= max_distance:
                    connected.add(target)
            except nx.NetworkXNoPath:
                continue
        
        return connected
    
    def sample_concept_combination(self, 
                                 num_concepts: int = 3, 
                                 diversity_weight: float = 0.7) -> List[CodeConcept]:
        """Sample a diverse combination of concepts for question generation."""
        if len(self.concepts) < num_concepts:
            return list(self.concepts.values())
        
        # Strategy: Mix random selection with diversity constraints
        selected_concepts = []
        selected_categories = set()
        selected_complexity = set()
        
        available_concepts = list(self.concepts.keys())
        random.shuffle(available_concepts)
        
        for concept_name in available_concepts:
            if len(selected_concepts) >= num_concepts:
                break
                
            concept = self.concepts[concept_name]
            
            # Diversity scoring
            diversity_score = 0
            if concept.category not in selected_categories:
                diversity_score += 1
            if concept.complexity_level not in selected_complexity:
                diversity_score += 1
            
            # Accept based on diversity weight and randomness
            accept_prob = diversity_weight * (diversity_score / 2) + (1 - diversity_weight) * random.random()
            
            if accept_prob > 0.3 or len(selected_concepts) == 0:
                selected_concepts.append(concept)
                selected_categories.add(concept.category)
                selected_complexity.add(concept.complexity_level)
        
        # Fill remaining slots randomly if needed
        while len(selected_concepts) < num_concepts and len(selected_concepts) < len(self.concepts):
            remaining = [c for name, c in self.concepts.items() 
                        if c not in selected_concepts]
            if remaining:
                selected_concepts.append(random.choice(remaining))
        
        return selected_concepts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_concepts": len(self.concepts),
            "categories": dict(self.concept_categories),
            "graph_density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph)
        }


class Level3QuestionGenerator:
    """
    Level-3 Question Generator that creates high-quality programming questions
    by combining concepts from multiple documents and leveraging knowledge graphs.
    """
    
    def __init__(self, llm_interface: LLMInterface, prompt1: str = "", prompt2: str = ""):
        self.llm = llm_interface
        self.prompt1 = prompt1 or self._default_concept_extraction_prompt()
        self.prompt2 = prompt2 or self._default_question_generation_prompt()
        self.knowledge_graph = KnowledgeGraph()
        self.processed_documents: List[str] = []
    
    def _default_concept_extraction_prompt(self) -> str:
        return """
        Extract key programming concepts from the following documentation.
        For each concept, provide:
        - name: Clear concept name
        - category: Type (data_structure, algorithm, design_pattern, syntax, etc.)
        - description: Brief explanation
        - complexity_level: basic/intermediate/advanced
        - related_concepts: List of related concepts
        - code_examples: Relevant code snippets
        
        Document: {document}
        """
    
    def _default_question_generation_prompt(self) -> str:
        return """
        Generate diverse, high-quality programming questions that combine the following concepts:
        {concepts}
        
        Create questions that:
        1. Require understanding of multiple concepts simultaneously
        2. Have practical, real-world applications
        3. Test both theoretical knowledge and implementation skills
        4. Vary in difficulty and question type (implementation, design, analysis)
        
        Generate 3-5 questions of different types.
        """
    
    def process_documents(self, documents: List[str]) -> None:
        """Process multiple reference documents to build the knowledge graph."""
        print(f"Processing {len(documents)} documents...")
        
        for i, document in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}")
            
            # Extract concepts using LLM
            raw_concepts = self.llm.extract_concepts(document, self.prompt1)
            
            # Convert to CodeConcept objects and add to knowledge graph
            for concept_data in raw_concepts:
                try:
                    concept = CodeConcept(
                        name=concept_data.get("name", ""),
                        category=concept_data.get("category", "unknown"),
                        description=concept_data.get("description", ""),
                        complexity_level=concept_data.get("complexity_level", "basic"),
                        related_concepts=concept_data.get("related_concepts", []),
                        code_examples=concept_data.get("code_examples", []),
                        source_document=f"doc_{i}"
                    )
                    self.knowledge_graph.add_concept(concept)
                except Exception as e:
                    print(f"Error processing concept: {e}")
                    continue
            
            self.processed_documents.append(f"doc_{i}")
        
        print(f"Knowledge graph built with {len(self.knowledge_graph.concepts)} concepts")
    
    def generate_questions(self, 
                         num_questions: int = 10, 
                         concepts_per_question: int = 3) -> List[Dict[str, Any]]:
        """Generate high-quality questions using the Level-3 method."""
        if not self.knowledge_graph.concepts:
            raise ValueError("No concepts available. Process documents first.")
        
        generated_questions = []
        
        for i in range(num_questions):
            try:
                # Random walk to sample concept combination
                concept_combination = self.knowledge_graph.sample_concept_combination(
                    num_concepts=concepts_per_question
                )
                
                if not concept_combination:
                    continue
                
                # Generate questions using LLM
                questions = self.llm.generate_questions(concept_combination, self.prompt2)
                
                # Package results
                for question in questions:
                    generated_questions.append({
                        "question": question,
                        "concepts_used": [c.name for c in concept_combination],
                        "concept_categories": [c.category for c in concept_combination],
                        "complexity_levels": [c.complexity_level for c in concept_combination],
                        "source_documents": list(set(c.source_document for c in concept_combination)),
                        "generation_method": "level-3"
                    })
                
            except Exception as e:
                print(f"Error generating question {i}: {e}")
                continue
        
        return generated_questions
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the constructed knowledge graph."""
        return self.knowledge_graph.get_stats()
    
    def export_knowledge_graph(self, filepath: str) -> None:
        """Export the knowledge graph for analysis or reuse."""
        graph_data = {
            "concepts": {name: {
                "category": concept.category,
                "description": concept.description,
                "complexity_level": concept.complexity_level,
                "related_concepts": concept.related_concepts,
                "source_document": concept.source_document
            } for name, concept in self.knowledge_graph.concepts.items()},
            "edges": list(self.knowledge_graph.graph.edges()),
            "stats": self.get_knowledge_graph_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Initialize the generator with a mock LLM
    mock_llm = MockLLM()
    generator = Level3QuestionGenerator(mock_llm)
    
    # Sample documents (in practice, these would be your reference documents)
    sample_documents = [
        "Documentation about data structures: arrays, linked lists, trees, graphs...",
        "Algorithm design patterns: divide and conquer, dynamic programming, greedy...", 
        "Object-oriented programming: inheritance, polymorphism, encapsulation..."
    ]
    
    # Process documents to build knowledge graph
    generator.process_documents(sample_documents)
    
    # Generate questions
    questions = generator.generate_questions(num_questions=5, concepts_per_question=2)
    
    # Display results
    print(f"\nGenerated {len(questions)} questions:")
    for i, q in enumerate(questions[:3], 1):  # Show first 3
        print(f"\nQuestion {i}:")
        print(f"Q: {q['question']}")
        print(f"Concepts: {', '.join(q['concepts_used'])}")
        print(f"Categories: {', '.join(q['concept_categories'])}")
    
    # Show knowledge graph stats
    print(f"\nKnowledge Graph Stats:")
    stats = generator.get_knowledge_graph_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")