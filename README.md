# code_rag_gen
Key Components:

1.CodeConcept: Data structure to represent programming concepts with categories like data structures, algorithms, design patterns, etc.
2.LLMInterface: Abstract interface for LLM interactions with two main methods:

extract_concepts() - uses prompt1 to extract concepts from documents
generate_questions() - uses prompt2 to generate questions from concept combinations


3.KnowledgeGraph: Manages the concept relationships using NetworkX:

Builds a graph of interconnected programming concepts
Supports concept sampling with diversity constraints
Provides graph statistics and analysis


4.Level3QuestionGenerator: Main class that orchestrates the entire process:

Processes multiple reference documents
Builds the knowledge graph from extracted concepts
Performs random walks to sample diverse concept combinations
Generates high-quality questions using the LLM



Key Features for Code Data:

Concept Categories: Tailored for programming (data_structure, algorithm, design_pattern, syntax)
Complexity Levels: Basic, intermediate, advanced
Diversity Sampling: Ensures questions combine concepts from different categories/complexity levels
Code Examples: Stores relevant code snippets with each concept
Multi-document Grounding: Combines concepts from multiple source documents

Usage

# Initialize with your LLM interface
generator = Level3QuestionGenerator(your_llm_interface, prompt1, prompt2)

# Process your reference documents
generator.process_documents(your_code_documents)

# Generate diverse, high-quality questions
questions = generator.generate_questions(num_questions=50, concepts_per_question=3)