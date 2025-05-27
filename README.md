# Level-3 Code Question Generator

A Python implementation of the Level-3 Question Generation methodology from "Scaling Laws of Synthetic Data for Language Models", adapted specifically for generating high-quality programming questions from code documentation.

## Overview

This project implements a sophisticated question generation system that:

1. **Extracts programming concepts** from multiple reference documents using LLMs
2. **Builds knowledge graphs** to understand relationships between concepts
3. **Performs intelligent sampling** to create diverse concept combinations
4. **Generates synthetic questions** that require understanding of multiple interconnected concepts

The Level-3 approach ensures questions are more complex and realistic than those generated from single documents, promoting diversity and scalability in synthetic data generation.

## Key Features

- 🧠 **Multi-document Knowledge Graph**: Constructs interconnected concept networks from multiple code documentation sources
- 🎯 **Intelligent Concept Sampling**: Uses random walks with diversity constraints to select concept combinations  
- 🔄 **Modular LLM Integration**: Abstract interface allows easy integration with different language models
- 📊 **Comprehensive Analytics**: Provides detailed statistics about generated questions and knowledge graphs
- 🐍 **Code-Focused**: Specifically designed for programming concepts (data structures, algorithms, OOP, etc.)

## Architecture

```
Reference Documents → Concept Extraction → Knowledge Graph → Random Sampling → Question Generation
                           ↓                    ↓               ↓                  ↓
                      [LLM + Prompt1]    [NetworkX Graph]  [Diversity Logic]  [LLM + Prompt2]
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd level3-question-generator

# Install required dependencies
pip install networkx
pip install numpy
pip install matplotlib  # optional, for graph visualization

# For testing with sample documents
pip install -r requirements.txt  # if available
```

## Quick Start

```python
from level3_generator import Level3QuestionGenerator, MockLLM
from sample_documents import get_sample_documents

# Initialize with your LLM interface (or use MockLLM for testing)
llm = MockLLM()  # Replace with your actual LLM interface
generator = Level3QuestionGenerator(llm)

# Load sample documents
documents = get_sample_documents()

# Process documents to build knowledge graph
generator.process_documents(documents)

# Generate high-quality questions
questions = generator.generate_questions(
    num_questions=20, 
    concepts_per_question=3
)

# Display results
for i, q in enumerate(questions[:5], 1):
    print(f"\nQuestion {i}:")
    print(f"Q: {q['question']}")
    print(f"Concepts: {', '.join(q['concepts_used'])}")
    print(f"Categories: {', '.join(q['concept_categories'])}")

# View knowledge graph statistics
stats = generator.get_knowledge_graph_stats()
print(f"\nKnowledge Graph Stats: {stats}")
```

## Core Components

### 1. CodeConcept
Data structure representing programming concepts:
```python
@dataclass
class CodeConcept:
    name: str                    # e.g., "Binary Search Tree"
    category: str               # e.g., "data_structure"
    description: str            # Concept explanation
    complexity_level: str       # "basic", "intermediate", "advanced"
    related_concepts: List[str] # Connected concepts
    code_examples: List[str]    # Relevant code snippets
    source_document: str        # Origin document
```

### 2. LLMInterface
Abstract interface for language model integration:
```python
class LLMInterface(ABC):
    @abstractmethod
    def extract_concepts(self, document: str, prompt: str) -> List[Dict]:
        """Extract concepts using prompt1"""
        pass
    
    @abstractmethod  
    def generate_questions(self, concepts: List[CodeConcept], prompt: str) -> List[str]:
        """Generate questions using prompt2"""
        pass
```

### 3. KnowledgeGraph
Manages concept relationships and sampling:
```python
class KnowledgeGraph:
    def add_concept(self, concept: CodeConcept) -> None
    def sample_concept_combination(self, num_concepts: int) -> List[CodeConcept]
    def get_connected_concepts(self, concept_name: str) -> Set[str]
    def get_stats(self) -> Dict[str, Any]
```

### 4. Level3QuestionGenerator
Main orchestrator class:
```python
class Level3QuestionGenerator:
    def process_documents(self, documents: List[str]) -> None
    def generate_questions(self, num_questions: int) -> List[Dict]
    def export_knowledge_graph(self, filepath: str) -> None
```

## Sample Documents

The project includes comprehensive sample documents covering:

1. **Data Structures** (`DATA_STRUCTURES_DOC`)
   - Arrays, Linked Lists, Binary Trees, Hash Tables
   - Complete implementations with time/space complexity analysis

2. **Algorithms** (`ALGORITHMS_DOC`)  
   - Divide & Conquer, Dynamic Programming, Greedy Algorithms
   - Graph algorithms (DFS, BFS, Dijkstra)

3. **Object-Oriented Programming** (`OOP_DOC`)
   - Classes, Inheritance, Polymorphism, Encapsulation
   - Design Patterns (Singleton, Factory, Observer)

4. **Advanced Python** (`ADVANCED_PYTHON_DOC`)
   - Decorators, Context Managers, Generators, Metaclasses
   - Async Programming, Type Hints

## Customization

### Custom LLM Integration
Replace `MockLLM` with your actual language model:

```python
class MyLLMInterface(LLMInterface):
    def __init__(self, api_key, model_name):
        self.client = initialize_llm_client(api_key, model_name)
    
    def extract_concepts(self, document: str, prompt: str) -> List[Dict]:
        response = self.client.complete(prompt.format(document=document))
        return parse_concepts_response(response)
    
    def generate_questions(self, concepts: List[CodeConcept], prompt: str) -> List[str]:
        concepts_text = format_concepts_for_prompt(concepts)
        response = self.client.complete(prompt.format(concepts=concepts_text))
        return parse_questions_response(response)
```

## Others

### Tips

如果误提交了 __pycache__ 怎么办

1. 从 Git 中删除缓存目录（保留本地文件）：
    ```bash
    git rm -r --cached */__pycache__  # 删除所有子目录中的 __pycache__
    git commit -m "Remove __pycache__ from repo"
    ```
2. 确保 .gitignore 已正确配置，避免再次误提交。