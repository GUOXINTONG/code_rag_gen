from data_structures import MockLLM
from level3_generator import Level3QuestionGenerator
from sample_documents import get_sample_documents

# Initialize with your LLM interface (or use MockLLM for testing)
llm = MockLLM()  # Replace with your actual LLM interface
generator = Level3QuestionGenerator(llm)

# Load sample documents
documents = get_sample_documents()

# Process documents to build knowledge graph
generator.process_documents(documents, ["ds_algo", "oop", "system_design"])

# Generate questions
questions = generator.generate_questions(num_questions=20, concepts_per_question=3)

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