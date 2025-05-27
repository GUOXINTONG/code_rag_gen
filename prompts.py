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
        return """Here is an article crawled from the web, which our classifier has identified as having significant educational value for students learning programming and software development.

Your task is to analyze this article and extract educational materials, specifically focusing on topics and key concepts that can enhance students' understanding of programming and improve their coding and problem-solving skills.

Pay special attention to uncommon but important programming concepts that are crucial for a deeper understanding of software development.

## Tasks

1. **Determine Educational Level:**
   - Identify the appropriate educational level for the article based on its content. Choose from the following options:
   - Beginner (New to programming)
   - Intermediate (Some programming experience)
   - Advanced (Experienced programmer)
   - Professional (Industry-level expertise)
   - Competition (Competitive programming/algorithmic contests)
   - Other

2. **Identify Subject Area:**
   - Specify the primary subject area of programming to which the article belongs (e.g., Data Structures, Algorithms, Web Development, Machine Learning, System Design, etc.).

3. **Extract Topics and Key Concepts:**
   - **Topics:**
     - List **1 to 5** main topics covered in the article.
     - Use terms commonly recognized in academia or industry.
   - **Key Concepts:**
     - For each identified topic, list **5 to 20** related key concepts.
     - Ensure these concepts are clearly articulated using standard academic or industry terms.

## Guidelines:
- **Terminology:** Use precise and widely recognized academic or industry terminology for subjects, topics, and key concepts to maintain consistency and clarity.
- **Educational Level Selection:** If appropriate, restrict the educational level to one of the following: "Beginner", "Intermediate", "Advanced", "Professional", or "Competition" to ensure accurate categorization.

## Text
{{ text }}

## Output Format
<level>Educational Level</level>
<subject>Subject Area</subject>
<topic>Topics:
1. topic 1
2. topic 2
</topic>
<key_concept>
Key Concepts:
1. topic 1:
   1.1. key concept
   1.2. key concept
   ...
2. topic 2:
   2.1. key concept
   ...
</key_concept>

## Output"""

    @staticmethod
    def get_question_generation_prompt() -> str:
        """
        Prompt for generating programming questions from concept combinations.
        Adapted from the original math question generation prompt.
        """
        return """As a senior **programming** instructor, your task is to create **diverse and challenging coding and system design questions** based on provided topics and knowledge points. These questions should demonstrate the application of the provided topics and key concepts while enhancing students' programming, problem-solving, and critical-thinking skills. Ensure that questions are **non-redundant**, precise, and engaging.

You will be provided with a list of key programming concepts spanning various topics and two relevant reference materials.

### Guidelines for Creating Diverse and Challenging Programming Questions:

1. **Concept Selection**:
   - Adhere to the Provided Topics: Ensure that each question aligns closely with the given topics.
   - Incorporate Multiple Concepts about Different Topics: Each question should encompass **2 or 3 key concepts about different programming topics**.
   - Ensure **broad coverage** of the provided concepts across the generated questions, avoiding **over-reliance** on simple or common applications of concepts.
   - Avoid repeating the same **concept combinations** or **implementation approach** across questions.

2. **Diversity and Challenge**:
   - Encourage **Cross-Topic Thinking**: By integrating concepts about different programming topics, questions will promote holistic understanding and application of programming principles.
   - **Leverage the Two Reference Materials**: The combination of both reference materials provides a **broader and more diverse context**, allowing for the creation of questions that explore a wider range of scenarios and applications. Use this to generate questions that challenge students in both familiar and novel contexts.
   - Ensure questions explore **different perspectives** and **applications** of the key concepts. Ensure each question is **sufficiently challenging** (e.g., requiring multi-step implementation, integrating real-world scenarios, involving system design or advanced algorithmic reasoning).

3. **Clarity and Precision**:
   - Use precise and unambiguous language.
   - Write all code snippets or pseudocode clearly and consistently.
   - Clearly state all assumptions, constraints, or requirements.
   - Specify expected input/output formats when applicable.

4. **Reference Material**:
   - Use the provided **reference articles about different topics** as sources of inspiration for generating **unique, diverse, and challenging questions**.
   - The combination of these two materials allows you to create questions with **more varied perspectives, contexts, and applications**, which can help test students' abilities to apply concepts in different situations.
   - The reference material is intended to:
     - Supplement the concept list by introducing **novel perspectives**, **contexts**, or **applications**.
     - Help create questions that are **more complex, much harder, and uncommon** in traditional teaching scenarios.
     - Serve as a resource to craft **real-world scenarios** or **system design challenges** beyond the given concepts.

5. **Output Diversity**:
   - Create between **1 to 3 questions**.
   - Ensure each question is unique in **structure**, **approach**, and **concept usage**.
   - Minimize the use of **sub-questions**, unless they are essential to the problem's complexity.
   - Questions can include: implementation challenges, system design problems, algorithm optimization, code review scenarios, or architectural decisions.

### Question Types (choose appropriate ones):
- **Implementation**: Write code to solve a specific problem
- **System Design**: Design a system or architecture
- **Algorithm Analysis**: Analyze time/space complexity or optimize algorithms  
- **Code Review**: Identify issues or improvements in given code
- **Architecture**: Make architectural decisions and trade-offs
- **Debugging**: Find and fix bugs in code
- **Performance**: Optimize code for better performance

### Inputs:
- **Article**:
{{ text }}

- **Concept List**:
{{ concept }}

#### Output Format:
<Q1>
Selected Concepts: [Only insert 2-3 concepts here]
Question: [Only insert question here]
</Q1>

<Q2>
Selected Concepts: [Only insert 2-3 concepts here]
Question: [Only insert question here]
</Q2>"""

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