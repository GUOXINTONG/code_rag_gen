"""
Level-3 Question Generator for code data.
Main orchestrator that combines all components to generate high-quality programming questions.
"""

import re
import random
from typing import List, Dict, Any, Optional
from data_structures import CodeConcept, ExtractedContent, GeneratedQuestion, LLMInterface
from knowledge_graph import KnowledgeGraph
from prompts import PromptTemplates


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
                            description=f"Programming concept related to {topic}",
                            complexity_level=self._determine_complexity_level(concept_name, extracted_content.educational_level),
                            related_concepts=related_concepts,
                            code_examples=[],  # Will be populated if needed
                            source_document=doc_id,
                            educational_level=extracted_content.educational_level
                        )
                        
                        self.knowledge_graph.add_concept(concept)
                
                self.processed_documents.append(doc_id)
                
            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")
                continue
        
        print(f"Knowledge graph built with {len(self.knowledge_graph.concepts)} concepts")
        print(f"Graph statistics: {self.knowledge_graph.get_stats()}")
    
    def _categorize_concept(self, concept_name: str, topic: str) -> str:
        """Categorize a concept based on its name and topic context."""
        concept_lower = concept_name.lower()
        topic_lower = topic.lower()
        
        # Define keyword mappings for categories
        category_keywords = {
            'data_structure': [
                'array', 'list', 'stack', 'queue', 'tree', 'graph', 'hash', 'heap',
                'linked list', 'binary tree', 'trie', 'set', 'map', 'dictionary'
            ],
            'algorithm': [
                'sort', 'search', 'traversal', 'recursion', 'dynamic programming',
                'greedy', 'divide and conquer', 'backtracking', 'optimization'
            ],
            'design_pattern': [
                'singleton', 'factory', 'observer', 'strategy', 'decorator',
                'adapter', 'facade', 'mvc', 'mvp', 'pattern'
            ],
            'programming_paradigm': [
                'oop', 'functional', 'procedural', 'declarative', 'imperative',
                'object-oriented', 'inheritance', 'polymorphism', 'encapsulation'
            ],
            'system_design': [
                'architecture', 'scalability', 'distributed', 'microservices',
                'database', 'caching', 'load balancing', 'api'
            ],
            'framework': [
                'framework', 'library', 'react', 'angular', 'vue', 'django',
                'flask', 'spring', 'express', 'rails'
            ],
            'language_feature': [
                'syntax', 'grammar', 'operator', 'keyword', 'function',
                'class', 'module', 'package', 'namespace'
            ]
        }
        
        # Check concept name and topic for category keywords
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in concept_lower or keyword in topic_lower:
                    return category
        
        # Default category
        return 'general_programming'
    
    def _determine_complexity_level(self, concept_name: str, educational_level: str) -> str:
        """Determine complexity level of a concept."""
        concept_lower = concept_name.lower()
        
        # Advanced concepts
        advanced_keywords = [
            'dynamic programming', 'graph algorithms', 'system design',
            'distributed systems', 'concurrency', 'parallel', 'optimization',
            'advanced', 'complex', 'efficient'
        ]
        
        # Beginner concepts
        beginner_keywords = [
            'basic', 'introduction', 'fundamentals', 'simple', 'elementary',
            'hello world', 'variables', 'loops', 'conditions'
        ]
        
        # Check keywords
        for keyword in advanced_keywords:
            if keyword in concept_lower:
                return 'advanced'
        
        for keyword in beginner_keywords:
            if keyword in concept_lower:
                return 'beginner'
        
        # Map educational level to complexity
        level_mapping = {
            'beginner': 'beginner',
            'intermediate': 'intermediate', 
            'advanced': 'advanced',
            'professional': 'advanced',
            'competition': 'expert'
        }
        
        return level_mapping.get(educational_level, 'intermediate')
    
    def generate_questions(self, 
                         num_questions: int = 10, 
                         concepts_per_question: int = 3,
                         diversity_weight: float = 0.7,
                         prefer_connected: bool = True) -> List[GeneratedQuestion]:
        """Generate high-quality questions using the Level-3 method."""
        if not self.knowledge_graph.concepts:
            raise ValueError("No concepts available. Process documents first.")
        
        generated_questions = []
        
        print(f"Generating {num_questions} questions with {concepts_per_question} concepts each...")
        
        for i in range(num_questions):
            try:
                # Sample concept combination using knowledge graph
                concept_combination = self.knowledge_graph.sample_concept_combination(
                    num_concepts=concepts_per_question,
                    diversity_weight=diversity_weight,
                    prefer_connected=prefer_connected
                )
                
                if not concept_combination:
                    print(f"Warning: Could not sample concepts for question {i+1}")
                    continue
                
                # Prepare reference materials from source documents
                source_docs = list(set(c.source_document for c in concept_combination))
                reference_materials = [
                    self.document_contents.get(doc_id, "") 
                    for doc_id in source_docs
                ]
                
                # Create concept list for prompt
                concept_list = self._format_concepts_for_prompt(concept_combination)
                
                # Generate question prompt
                reference_text = "\n\n".join([
                    f"Document {i+1}:\n{content[:1000]}..." 
                    for i, content in enumerate(reference_materials) if content
                ])
                
                prompt = self.prompt_templates.format_question_generation_prompt(
                    reference_text, concept_list
                )
                
                # Generate questions using LLM
                questions = self.llm.generate_questions(
                    concept_combination, reference_materials, prompt
                )
                
                generated_questions.extend(questions)
                
                if (i + 1) % 5 == 0:
                    print(f"Generated questions for {i + 1}/{num_questions} iterations")
                
            except Exception as e:
                print(f"Error generating question {i+1}: {e}")
                continue
        
        print(f"Successfully generated {len(generated_questions)} questions")
        return generated_questions
    
    def _format_concepts_for_prompt(self, concepts: List[CodeConcept]) -> str:
        """Format concepts for inclusion in the question generation prompt."""
        concept_descriptions = []
        
        for concept in concepts:
            description = f"- {concept.name} ({concept.category}): {concept.description}"
            if concept.related_concepts:
                related = ", ".join(concept.related_concepts[:3])
                description += f" [Related: {related}]"
            concept_descriptions.append(description)
        
        return "\n".join(concept_descriptions)
    
    def generate_questions_by_category(self, 
                                     category: str, 
                                     num_questions: int = 5,
                                     concepts_per_question: int = 2) -> List[GeneratedQuestion]:
        """Generate questions focused on a specific category."""
        category_concepts = self.knowledge_graph.get_concepts_by_category(category)
        
        if len(category_concepts) < concepts_per_question:
            raise ValueError(f"Not enough concepts in category '{category}'. "
                           f"Found {len(category_concepts)}, need at least {concepts_per_question}")
        
        questions = []
        
        for i in range(num_questions):
            try:
                # Sample from category concepts
                selected_concepts = random.sample(category_concepts, 
                                                min(concepts_per_question, len(category_concepts)))
                
                # Add some concepts from other categories for diversity
                other_concepts = [c for c in self.knowledge_graph.concepts.values() 
                                if c.category != category]
                if other_concepts:
                    additional_count = max(0, concepts_per_question - len(selected_concepts))
                    if additional_count > 0:
                        additional = random.sample(other_concepts, 
                                                 min(additional_count, len(other_concepts)))
                        selected_concepts.extend(additional)
                
                # Generate questions
                source_docs = list(set(c.source_document for c in selected_concepts))
                reference_materials = [
                    self.document_contents.get(doc_id, "") 
                    for doc_id in source_docs
                ]
                
                concept_list = self._format_concepts_for_prompt(selected_concepts)
                reference_text = "\n\n".join([
                    f"Document {i+1}:\n{content[:1000]}..." 
                    for i, content in enumerate(reference_materials) if content
                ])
                
                prompt = self.prompt_templates.format_question_generation_prompt(
                    reference_text, concept_list
                )
                
                category_questions = self.llm.generate_questions(
                    selected_concepts, reference_materials, prompt
                )
                
                questions.extend(category_questions)
                
            except Exception as e:
                print(f"Error generating category question {i+1}: {e}")
                continue
        
        return questions
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        return self.knowledge_graph.get_stats()
    
    def export_knowledge_graph(self, filepath: str, format: str = "json") -> None:
        """Export the knowledge graph for analysis or reuse."""
        self.knowledge_graph.export_graph(filepath, format)
    
    def load_knowledge_graph(self, filepath: str) -> None:
        """Load a previously saved knowledge graph."""
        self.knowledge_graph.load_graph(filepath)
        # Update processed documents list
        self.processed_documents = list(set(
            concept.source_document 
            for concept in self.knowledge_graph.concepts.values()
        ))
    
    def get_concept_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Get clusters of related concepts."""
        clusters = self.knowledge_graph.find_concept_clusters(min_cluster_size)
        return [list(cluster) for cluster in clusters]
    
    def generate_questions_from_cluster(self, 
                                      cluster_concepts: List[str], 
                                      num_questions: int = 3) -> List[GeneratedQuestion]:
        """Generate questions from a specific concept cluster."""
        concepts = [self.knowledge_graph.concepts[name] 
                   for name in cluster_concepts 
                   if name in self.knowledge_graph.concepts]
        
        if len(concepts) < 2:
            raise ValueError("Need at least 2 valid concepts to generate questions")
        
        questions = []
        
        for i in range(num_questions):
            try:
                # Sample from cluster concepts
                sample_size = min(3, len(concepts))
                selected_concepts = random.sample(concepts, sample_size)
                
                # Generate questions
                source_docs = list(set(c.source_document for c in selected_concepts))
                reference_materials = [
                    self.document_contents.get(doc_id, "") 
                    for doc_id in source_docs
                ]
                
                concept_list = self._format_concepts_for_prompt(selected_concepts)
                reference_text = "\n\n".join([
                    f"Document {i+1}:\n{content[:1000]}..." 
                    for i, content in enumerate(reference_materials) if content
                ])
                
                prompt = self.prompt_templates.format_question_generation_prompt(
                    reference_text, concept_list
                )
                
                cluster_questions = self.llm.generate_questions(
                    selected_concepts, reference_materials, prompt
                )
                
                questions.extend(cluster_questions)
                
            except Exception as e:
                print(f"Error generating cluster question {i+1}: {e}")
                continue
        
        return questions
    
    def analyze_concept_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between concepts in the knowledge graph."""
        stats = self.get_knowledge_graph_stats()
        
        # Find most connected concepts
        most_connected = stats.get('most_connected_concepts', [])
        
        # Analyze cross-category connections
        cross_category_edges = 0
        total_edges = 0
        
        for edge in self.knowledge_graph.graph.edges():
            concept1 = self.knowledge_graph.concepts[edge[0]]
            concept2 = self.knowledge_graph.concepts[edge[1]]
            
            total_edges += 1
            if concept1.category != concept2.category:
                cross_category_edges += 1
        
        cross_category_ratio = cross_category_edges / total_edges if total_edges > 0 else 0
        
        # Find concept clusters
        clusters = self.get_concept_clusters()
        
        return {
            'total_concepts': len(self.knowledge_graph.concepts),
            'most_connected_concepts': most_connected,
            'cross_category_connection_ratio': cross_category_ratio,
            'concept_clusters': len(clusters),
            'largest_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0,
            'categories': stats.get('categories', {}),
            'educational_levels': stats.get('educational_levels', {})
        }


# Example usage and testing
if __name__ == "__main__":
    from data_structures import MockLLM
    
    # Initialize generator
    mock_llm = MockLLM()
    generator = Level3QuestionGenerator(mock_llm)
    
    # Sample documents
    sample_documents = [
        """
        Data Structures and Algorithms
        
        This document covers fundamental data structures including arrays, linked lists, 
        stacks, queues, trees, and graphs. We also explore sorting algorithms like 
        quicksort and mergesort, as well as searching techniques.
        
        Key topics:
        - Binary Search Trees
        - Graph Traversal (DFS, BFS)  
        - Dynamic Programming
        - Hash Tables
        """,
        """
        Object-Oriented Programming
        
        Learn about OOP principles including encapsulation, inheritance, and polymorphism.
        Covers design patterns like Singleton, Factory, and Observer patterns.
        
        Key topics:
        - Class Design
        - Inheritance Hierarchies
        - Design Patterns
        - SOLID Principles
        """,
        """
        System Design and Architecture
        
        Covers scalable system design, distributed systems, microservices architecture,
        database design, caching strategies, and load balancing.
        
        Key topics:
        - Microservices Architecture
        - Database Sharding
        - Caching Strategies
        - Load Balancing
        """
    ]
    
    # Process documents
    generator.process_documents(sample_documents, ["ds_algo", "oop", "system_design"])
    
    # Generate questions
    questions = generator.generate_questions(num_questions=5, concepts_per_question=3)
    
    # Display results
    print(f"\nGenerated {len(questions)} questions:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\nQuestion {i}:")
        print(f"Q: {q.question}")
        print(f"Concepts: {', '.join(q.selected_concepts)}")
        print(f"Categories: {', '.join(set(q.concept_categories))}")
        print(f"Sources: {', '.join(q.source_documents)}")
    
    # Show analysis
    print(f"\nConcept Relationship Analysis:")
    analysis = generator.analyze_concept_relationships()
    for key, value in analysis.items():
        print(f"{key}: {value}")

'''
# Actual usage
if __name__ == "__main__":
    # Initialize with your LLM
    generator = Level3QuestionGenerator(your_llm_interface)

    # Process your code documents
    generator.process_documents(documents, ["doc1", "doc2", "doc3"])

    # Generate diverse questions
    questions = generator.generate_questions(
        num_questions=20, 
        concepts_per_question=3,
        diversity_weight=0.7
    )

    # Generate category-specific questions
    ds_questions = generator.generate_questions_by_category(
        "data_structure", 
        num_questions=10
    )
'''