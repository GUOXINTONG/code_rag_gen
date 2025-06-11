import random
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import itertools

class Domain(Enum):
    BASIC_PROGRAMMING = "BP"
    ADVANCED_PROGRAMMING = "AP"
    SOFTWARE_ENGINEERING = "SE"
    DATA_ANALYSIS = "DP"
    MATHEMATICS = "MA"
    DESKTOP_WEB_DEV = "DW"
    MACHINE_LEARNING = "ML"
    SCIENTIFIC_COMPUTING = "SC"
    DATABASE = "DB"
    MULTIMEDIA = "MM"
    OPERATING_SYSTEM = "OS"
    OTHERS = "OT"

class Complexity(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class CodeScenario:
    domain: Domain
    complexity: Complexity
    question: str
    answer: str
    context: str
    tags: List[str]
    language: str
    realistic_context: str

class SyntheticCodeGenerator:
    def __init__(self):
        self.templates = self._initialize_templates()
        self.contexts = self._initialize_contexts()
        self.code_patterns = self._initialize_code_patterns()
        
    def _initialize_templates(self) -> Dict[Domain, List[Dict]]:
        """Initialize question templates for each domain"""
        return {
            Domain.BASIC_PROGRAMMING: [
                {
                    "template": "I need to {task} in {language}. {context_detail}",
                    "tasks": ["parse command line arguments", "read a configuration file", 
                             "validate user input", "format output data", "handle exceptions"],
                    "complexity_modifiers": {
                        Complexity.BEGINNER: "Can you show me a simple example?",
                        Complexity.INTERMEDIATE: "What's the best practice approach?",
                        Complexity.ADVANCED: "How can I make this more robust and efficient?",
                        Complexity.EXPERT: "What are the edge cases and performance considerations?"
                    }
                },
                {
                    "template": "How do I implement {pattern} in {language} for {use_case}?",
                    "patterns": ["error handling", "logging", "configuration management", 
                               "input validation", "data transformation"],
                    "use_cases": ["a CLI tool", "a batch processor", "a file converter", 
                                "a data validator", "a report generator"]
                }
            ],
            
            Domain.ADVANCED_PROGRAMMING: [
                {
                    "template": "I'm building {system_type} and need to implement {feature}. {constraint}",
                    "system_types": ["a distributed system", "a high-performance application", 
                                   "a microservice", "a plugin architecture", "a framework"],
                    "features": ["connection pooling", "circuit breaker pattern", "caching layer",
                               "rate limiting", "async processing", "state management"],
                    "constraints": ["It needs to handle 10k+ concurrent users.",
                                  "Memory usage must be minimal.",
                                  "It should be thread-safe.",
                                  "Performance is critical.",
                                  "It must be easily testable."]
                },
                {
                    "template": "How can I optimize {component} in my {application_type}? {current_issue}",
                    "components": ["database queries", "memory allocation", "network calls",
                                 "algorithm performance", "data structures"],
                    "application_types": ["web service", "desktop application", "mobile app",
                                        "game engine", "data pipeline"],
                    "current_issues": ["It's becoming a bottleneck.", "Memory leaks are occurring.",
                                     "Response times are too slow.", "CPU usage is too high.",
                                     "It doesn't scale well."]
                }
            ],
            
            Domain.SOFTWARE_ENGINEERING: [
                {
                    "template": "I need to {task} for a {project_type}. {requirement}",
                    "tasks": ["set up CI/CD pipeline", "implement automated testing",
                             "design system architecture", "refactor legacy code",
                             "implement logging and monitoring"],
                    "project_types": ["e-commerce platform", "content management system",
                                    "mobile application", "IoT system", "financial application"],
                    "requirements": ["The team has 5 developers.", "We use agile methodology.",
                                   "Security is paramount.", "We need 99.9% uptime.",
                                   "The system must be maintainable."]
                },
                {
                    "template": "How do I implement {pattern} in {context}? {additional_info}",
                    "patterns": ["dependency injection", "repository pattern", "factory pattern",
                               "observer pattern", "strategy pattern", "decorator pattern"],
                    "contexts": ["a REST API", "a desktop application", "a web framework",
                               "a game engine", "a data processing pipeline"],
                    "additional_info": ["I'm using {language}.", "It should be testable.",
                                      "Performance is important.", "It needs to be extensible.",
                                      "Multiple teams will use it."]
                }
            ],
            
            Domain.DATA_ANALYSIS: [
                {
                    "template": "I have {data_type} and need to {analysis_task}. {challenge}",
                    "data_types": ["sales data from multiple CSV files", "user behavior logs",
                                 "sensor data from IoT devices", "financial transaction records",
                                 "social media interaction data"],
                    "analysis_tasks": ["identify trends and patterns", "detect anomalies",
                                     "create predictive models", "generate automated reports",
                                     "perform statistical analysis"],
                    "challenges": ["The data is messy and inconsistent.", "There are missing values.",
                                 "The dataset is very large.", "Real-time processing is needed.",
                                 "The data comes from multiple sources."]
                },
                {
                    "template": "How can I {task} using {tool} for {domain}?",
                    "tasks": ["clean and preprocess data", "create interactive visualizations",
                             "implement data pipelines", "perform feature engineering",
                             "optimize query performance"],
                    "tools": ["pandas and numpy", "SQL and database views", "Apache Spark",
                            "matplotlib and seaborn", "scikit-learn"],
                    "domains": ["customer analytics", "financial analysis", "healthcare data",
                              "marketing campaigns", "operational metrics"]
                }
            ],
            
            Domain.MATHEMATICS: [
                {
                    "template": "I need to implement {algorithm} for {application}. {constraint}",
                    "algorithms": ["numerical integration", "optimization algorithms",
                                 "statistical calculations", "linear algebra operations",
                                 "probability distributions"],
                    "applications": ["scientific simulation", "financial modeling",
                                   "image processing", "game physics", "data analysis"],
                    "constraints": ["Accuracy is critical.", "Performance must be optimized.",
                                  "It should handle edge cases.", "Memory efficiency is important.",
                                  "It needs to be numerically stable."]
                },
                {
                    "template": "How do I solve {problem_type} computationally? {context}",
                    "problem_types": ["system of linear equations", "curve fitting",
                                    "root finding", "differential equations",
                                    "fourier transforms"],
                    "contexts": ["I'm working with large datasets.", "Precision is important.",
                               "Real-time processing is needed.", "The solution must be robust.",
                               "I need to handle multiple variables."]
                }
            ],
            
            Domain.DESKTOP_WEB_DEV: [
                {
                    "template": "I'm building {app_type} and need to {feature}. {requirement}",
                    "app_types": ["a desktop application", "a web application", "a mobile app",
                                "a browser extension", "a progressive web app"],
                    "features": ["implement user authentication", "create responsive UI",
                               "handle file uploads", "implement real-time updates",
                               "integrate with external APIs"],
                    "requirements": ["It should work offline.", "The UI must be accessible.",
                                   "Performance is crucial.", "It needs to be secure.",
                                   "Cross-platform compatibility is needed."]
                },
                {
                    "template": "How can I implement {component} in {framework}? {consideration}",
                    "components": ["state management", "routing", "form validation",
                                 "data binding", "component communication"],
                    "frameworks": ["React", "Vue.js", "Angular", "Electron", "Flutter"],
                    "considerations": ["The app will have complex state.", "SEO is important.",
                                     "Bundle size should be minimal.", "TypeScript is preferred.",
                                     "Testing is a priority."]
                }
            ],
            
            Domain.MACHINE_LEARNING: [
                {
                    "template": "I need to build {model_type} for {use_case}. {data_info}",
                    "model_types": ["a classification model", "a regression model",
                                  "a clustering algorithm", "a recommendation system",
                                  "an anomaly detection system"],
                    "use_cases": ["predicting customer churn", "image recognition",
                                "natural language processing", "time series forecasting",
                                "fraud detection"],
                    "data_info": ["I have limited training data.", "The data is imbalanced.",
                                "Real-time inference is needed.", "The model must be interpretable.",
                                "I need to handle streaming data."]
                },
                {
                    "template": "How do I {task} for my ML pipeline? {challenge}",
                    "tasks": ["preprocess and feature engineer", "handle model deployment",
                             "implement model monitoring", "optimize hyperparameters",
                             "handle data drift"],
                    "challenges": ["The model needs to be updated regularly.",
                                 "Inference latency must be low.",
                                 "The system must be scalable.",
                                 "I need to track model performance.",
                                 "Data privacy is a concern."]
                }
            ],
            
            Domain.SCIENTIFIC_COMPUTING: [
                {
                    "template": "I need to {task} for {domain}. {requirement}",
                    "tasks": ["simulate physical processes", "solve numerical problems",
                             "implement parallel algorithms", "optimize computational performance",
                             "process large datasets"],
                    "domains": ["climate modeling", "fluid dynamics", "molecular dynamics",
                              "quantum mechanics", "structural analysis"],
                    "requirements": ["High precision is required.", "The computation is CPU-intensive.",
                                   "Memory usage must be optimized.", "Results must be reproducible.",
                                   "Parallel processing is needed."]
                },
                {
                    "template": "How can I implement {method} efficiently? {context}",
                    "methods": ["finite element analysis", "Monte Carlo simulation",
                              "Fast Fourier Transform", "gradient descent optimization",
                              "numerical differentiation"],
                    "contexts": ["I'm using HPC clusters.", "GPU acceleration is available.",
                               "The problem has multiple scales.", "Convergence is challenging.",
                               "Memory bandwidth is limited."]
                }
            ],
            
            Domain.DATABASE: [
                {
                    "template": "I need to {task} for {scenario}. {constraint}",
                    "tasks": ["optimize database queries", "design database schema",
                             "implement data migration", "set up database replication",
                             "handle concurrent transactions"],
                    "scenarios": ["an e-commerce platform", "a content management system",
                                "a real-time analytics system", "a multi-tenant application",
                                "a high-traffic web service"],
                    "constraints": ["Query performance is critical.", "Data consistency is important.",
                                  "The system must scale horizontally.", "Downtime must be minimal.",
                                  "ACID compliance is required."]
                },
                {
                    "template": "How do I implement {feature} in {database}? {use_case}",
                    "features": ["full-text search", "data partitioning", "indexing strategy",
                               "stored procedures", "triggers and constraints"],
                    "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch"],
                    "use_cases": ["For a search engine.", "To handle time-series data.",
                                "For user session management.", "To implement caching.",
                                "For analytics queries."]
                }
            ],
            
            Domain.MULTIMEDIA: [
                {
                    "template": "I need to {task} for {application}. {requirement}",
                    "tasks": ["process video files", "implement image recognition",
                             "handle audio streaming", "create image filters",
                             "implement video compression"],
                    "applications": ["a media player", "a video editing tool",
                                   "a streaming service", "a photo editing app",
                                   "a computer vision system"],
                    "requirements": ["Real-time processing is needed.", "Quality must be preserved.",
                                   "Memory usage should be optimized.", "Multiple formats must be supported.",
                                   "The solution should be cross-platform."]
                },
                {
                    "template": "How can I implement {feature} efficiently? {context}",
                    "features": ["image resizing and cropping", "audio format conversion",
                               "video transcoding", "face detection", "color correction"],
                    "contexts": ["Processing large batches of files.", "Working with high-resolution content.",
                               "Implementing on mobile devices.", "Using GPU acceleration.",
                               "Maintaining quality while reducing file size."]
                }
            ],
            
            Domain.OPERATING_SYSTEM: [
                {
                    "template": "I need to {task} for {system_type}. {challenge}",
                    "tasks": ["implement process scheduling", "manage memory allocation",
                             "handle file system operations", "implement device drivers",
                             "create system monitoring tools"],
                    "system_types": ["an embedded system", "a real-time system",
                                   "a distributed system", "a virtualized environment",
                                   "a containerized application"],
                    "challenges": ["Resource constraints are tight.", "Latency must be minimal.",
                                 "The system must be fault-tolerant.", "Security is paramount.",
                                 "Portability across platforms is needed."]
                },
                {
                    "template": "How do I implement {component} in {environment}? {consideration}",
                    "components": ["inter-process communication", "thread synchronization",
                                 "memory management", "network stack", "file system"],
                    "environments": ["Linux kernel space", "Windows system programming",
                                   "embedded systems", "real-time systems", "virtualized environments"],
                    "considerations": ["Performance is critical.", "Thread safety is required.",
                                     "The code must be portable.", "Error handling is important.",
                                     "Resource cleanup is essential."]
                }
            ],
            
            Domain.OTHERS: [
                {
                    "template": "I'm working on {project_type} and need to {task}. {context}",
                    "project_types": ["a blockchain application", "an IoT system",
                                    "a robotics project", "a game development",
                                    "a cryptocurrency trading bot"],
                    "tasks": ["implement smart contracts", "handle sensor data",
                             "create AI behavior", "optimize game performance",
                             "implement trading algorithms"],
                    "contexts": ["Security is the top priority.", "Real-time processing is needed.",
                               "The system must be distributed.", "User experience is important.",
                               "Scalability is a concern."]
                }
            ]
        }
    
    def _initialize_contexts(self) -> Dict[str, List[str]]:
        """Initialize realistic contexts for different scenarios"""
        return {
            "startup": ["We're a small startup with limited resources.",
                       "We need to move fast and iterate quickly.",
                       "Scalability is important for future growth.",
                       "We're building an MVP first."],
            "enterprise": ["This is for a large enterprise client.",
                          "Compliance and security are critical.",
                          "The solution must integrate with existing systems.",
                          "High availability is required."],
            "open_source": ["This will be an open-source project.",
                           "Community contributions are welcome.",
                           "Documentation is very important.",
                           "Cross-platform compatibility is needed."],
            "research": ["This is for a research project.",
                        "Reproducibility is important.",
                        "We need to publish our findings.",
                        "Novel approaches are encouraged."],
            "education": ["This is for educational purposes.",
                         "The code should be easy to understand.",
                         "Students will be extending this.",
                         "Best practices should be demonstrated."]
        }
    
    def _initialize_code_patterns(self) -> Dict[str, Dict]:
        """Initialize code patterns and solutions for different domains"""
        return {
            "error_handling": {
                "python": """
try:
    result = risky_operation()
    return result
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return default_value
finally:
    cleanup_resources()
""",
                "javascript": """
async function handleOperation() {
    try {
        const result = await riskyOperation();
        return result;
    } catch (error) {
        if (error instanceof SpecificError) {
            console.error('Operation failed:', error.message);
            throw error;
        }
        console.error('Unexpected error:', error);
        return defaultValue;
    }
}
""",
                "java": """
public Result handleOperation() {
    try {
        return performOperation();
    } catch (SpecificException e) {
        logger.error("Operation failed", e);
        throw new ServiceException("Service unavailable", e);
    } catch (Exception e) {
        logger.error("Unexpected error", e);
        return getDefaultResult();
    } finally {
        cleanupResources();
    }
}
"""
            },
            "async_patterns": {
                "python": """
import asyncio
from typing import List, Awaitable

async def process_batch(items: List[str]) -> List[str]:
    semaphore = asyncio.Semaphore(10)  # Limit concurrency
    
    async def process_item(item: str) -> str:
        async with semaphore:
            return await expensive_operation(item)
    
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions in results
    successful_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed: {result}")
        else:
            successful_results.append(result)
    
    return successful_results
""",
                "javascript": """
async function processBatch(items) {
    const concurrencyLimit = 10;
    const semaphore = new Semaphore(concurrencyLimit);
    
    const processItem = async (item) => {
        await semaphore.acquire();
        try {
            return await expensiveOperation(item);
        } finally {
            semaphore.release();
        }
    };
    
    const results = await Promise.allSettled(
        items.map(processItem)
    );
    
    return results
        .filter(result => result.status === 'fulfilled')
        .map(result => result.value);
}
"""
            }
        }
    
    def generate_scenario(self, domain: Domain, complexity: Complexity, 
                         language: str = None, context_type: str = None) -> CodeScenario:
        """Generate a single realistic coding scenario"""
        
        # Select template
        domain_templates = self.templates[domain]
        template_config = random.choice(domain_templates)
        
        # Generate question
        question = self._generate_question(template_config, complexity, language, context_type)
        
        # Generate answer
        answer = self._generate_answer(domain, complexity, language, question)
        
        # Generate context
        context = self._generate_context(domain, context_type)
        
        # Generate tags
        tags = self._generate_tags(domain, complexity, language)
        
        # Generate realistic context
        realistic_context = self._generate_realistic_context(domain, context_type)
        
        return CodeScenario(
            domain=domain,
            complexity=complexity,
            question=question,
            answer=answer,
            context=context,
            tags=tags,
            language=language or "python",
            realistic_context=realistic_context
        )
    
    def _generate_question(self, template_config: Dict, complexity: Complexity, 
                          language: str, context_type: str) -> str:
        """Generate a realistic question based on template"""
        template = template_config["template"]
        
        # Fill in template variables
        filled_template = template
        for key, values in template_config.items():
            if key != "template" and key != "complexity_modifiers":
                if isinstance(values, list):
                    value = random.choice(values)
                    filled_template = filled_template.replace(f"{{{key[:-1]}}}", value)
        
        # Add language if not specified
        if language and "{language}" in filled_template:
            filled_template = filled_template.replace("{language}", language)
        
        # Add complexity modifier if available
        if "complexity_modifiers" in template_config:
            modifier = template_config["complexity_modifiers"].get(complexity, "")
            if modifier:
                filled_template += f" {modifier}"
        
        return filled_template
    
    def _generate_answer(self, domain: Domain, complexity: Complexity, 
                        language: str, question: str) -> str:
        """Generate a comprehensive answer based on domain and complexity"""
        
        # This is a simplified version - in practice, you'd have more sophisticated
        # answer generation based on the specific question and domain
        
        base_answer = f"Here's a comprehensive solution for your {domain.value} question:\n\n"
        
        # Add code example based on domain
        code_example = self._generate_code_example(domain, complexity, language)
        base_answer += f"```{language or 'python'}\n{code_example}\n```\n\n"
        
        # Add explanation based on complexity
        explanation = self._generate_explanation(domain, complexity)
        base_answer += explanation
        
        # Add best practices
        best_practices = self._generate_best_practices(domain, complexity)
        base_answer += f"\n\n**Best Practices:**\n{best_practices}"
        
        # Add complexity-specific additions
        if complexity in [Complexity.ADVANCED, Complexity.EXPERT]:
            advanced_notes = self._generate_advanced_notes(domain, complexity)
            base_answer += f"\n\n**Advanced Considerations:**\n{advanced_notes}"
        
        return base_answer
    
    def _generate_code_example(self, domain: Domain, complexity: Complexity, language: str) -> str:
        """Generate domain-specific code examples"""
        
        # Simplified code generation - in practice, this would be much more sophisticated
        if domain == Domain.DATA_ANALYSIS:
            return """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def analyze_data(file_path):
    # Load and clean data
    df = pd.read_csv(file_path)
    df = df.dropna()
    
    # Perform analysis
    summary_stats = df.describe()
    correlation_matrix = df.corr()
    
    # Feature engineering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    return {
        'summary': summary_stats,
        'correlations': correlation_matrix,
        'scaled_features': scaled_features
    }
"""
        elif domain == Domain.DESKTOP_WEB_DEV:
            return """
// React component with hooks
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DataFetcher = ({ url }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(url);
                setData(response.data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
    }, [url]);
    
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;
    
    return (
        <div>
            <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
    );
};
"""
        else:
            # Generic example
            return """
def solve_problem(input_data):
    \"\"\"
    Solve the given problem efficiently
    \"\"\"
    try:
        # Process input
        processed_data = preprocess(input_data)
        
        # Apply algorithm
        result = apply_algorithm(processed_data)
        
        # Validate result
        if validate_result(result):
            return result
        else:
            raise ValueError("Invalid result")
            
    except Exception as e:
        logging.error(f"Error in solve_problem: {e}")
        raise
"""
    
    def _generate_explanation(self, domain: Domain, complexity: Complexity) -> str:
        """Generate explanation based on domain and complexity"""
        explanations = {
            Domain.DATA_ANALYSIS: "This solution handles data loading, cleaning, and analysis with proper error handling and scalable preprocessing.",
            Domain.DESKTOP_WEB_DEV: "This React component demonstrates modern hooks usage with proper state management and error handling.",
            Domain.MACHINE_LEARNING: "This implementation follows ML best practices with proper data preprocessing and model validation.",
            Domain.SOFTWARE_ENGINEERING: "This solution implements clean architecture principles with proper separation of concerns."
        }
        
        base_explanation = explanations.get(domain, "This solution follows best practices for the given domain.")
        
        if complexity == Complexity.EXPERT:
            base_explanation += " The implementation considers performance optimization, scalability, and enterprise-grade reliability."
        
        return base_explanation
    
    def _generate_best_practices(self, domain: Domain, complexity: Complexity) -> str:
        """Generate best practices based on domain"""
        practices = {
            Domain.DATA_ANALYSIS: "- Always validate input data\n- Handle missing values appropriately\n- Use vectorized operations for performance\n- Document data transformations",
            Domain.DESKTOP_WEB_DEV: "- Use proper state management\n- Implement error boundaries\n- Optimize for performance\n- Ensure accessibility",
            Domain.SOFTWARE_ENGINEERING: "- Follow SOLID principles\n- Implement comprehensive testing\n- Use dependency injection\n- Maintain clean code"
        }
        
        return practices.get(domain, "- Follow language-specific conventions\n- Implement proper error handling\n- Write comprehensive tests\n- Document your code")
    
    def _generate_advanced_notes(self, domain: Domain, complexity: Complexity) -> str:
        """Generate advanced considerations for expert-level scenarios"""
        return "- Consider performance implications at scale\n- Implement proper monitoring and logging\n- Plan for failure scenarios\n- Consider security implications\n- Design for maintainability and extensibility"
    
    def _generate_context(self, domain: Domain, context_type: str) -> str:
        """Generate context information"""
        if context_type and context_type in self.contexts:
            return random.choice(self.contexts[context_type])
        return "This is a typical development scenario requiring a practical solution."
    
    def _generate_realistic_context(self, domain: Domain, context_type: str) -> str:
        """Generate realistic project context"""
        contexts = {
            "startup": "Fast-growing startup, need to ship quickly but plan for scale",
            "enterprise": "Large enterprise client, security and compliance critical",
            "open_source": "Open source project, community-driven development",
            "research": "Academic research project, reproducibility important",
            "education": "Educational project, code clarity and documentation crucial"
        }
        
        return contexts.get(context_type, "Professional development project")
    
    def _generate_tags(self, domain: Domain, complexity: Complexity, language: str) -> List[str]:
        """Generate relevant tags for the scenario"""
        base_tags = [domain.value.lower(), complexity.value]
        
        if language:
            base_tags.append(language)
        
        # Add domain-specific tags
        domain_tags = {
            Domain.DATA_ANALYSIS: ["pandas", "numpy", "data-science", "analytics"],
            Domain.DESKTOP_WEB_DEV: ["react", "frontend", "ui", "web-development"],
            Domain.MACHINE_LEARNING: ["ml", "sklearn", "tensorflow", "data-science"],
            Domain.SOFTWARE_ENGINEERING: ["architecture", "design-patterns", "testing", "clean-code"]
        }
        
        if domain in domain_tags:
            base_tags.extend(random.sample(domain_tags[domain], 2))
        
        return base_tags
    
    def generate_batch(self, count: int, distribution: Dict[Domain, float] = None,
                      complexity_distribution: Dict[Complexity, float] = None,
                      languages: List[str] = None) -> List[CodeScenario]:
        """Generate a batch of code scenarios with specified distributions"""
        
        # Default distributions
        if distribution is None:
            distribution = {domain: 1.0/len(Domain) for domain in Domain}
        
        if complexity_distribution is None:
            complexity_distribution = {
                Complexity.BEGINNER: 0.3,
                Complexity.INTERMEDIATE: 0.4,
                Complexity.ADVANCED: 0.2,
                Complexity.EXPERT: 0.1
            }
        
        if languages is None:
            languages = ["python", "javascript", "java", "cpp", "go", "typescript"]
        
        scenarios = []
        
        for _ in range(count):
            # Select domain based on distribution
            domain = random.choices(
                list(distribution.keys()),
                weights=list(distribution.values())
            )[0]
            
            # Select complexity based on distribution
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values())
            )[0]
            
            # Select language
            language = random.choice(languages)
            
            # Select context type
            context_type = random.choice(["startup", "enterprise", "open_source", "research", "education"])
            
            scenario = self.generate_scenario(domain, complexity, language, context_type)
            scenarios.append(scenario)
        
        return scenarios
    
    def export_scenarios(self, scenarios: List[CodeScenario], filename: str):
        """Export scenarios to JSON file"""
        scenarios_dict = [asdict(scenario) for scenario in scenarios]
        
        # Convert enums to strings
        for scenario in scenarios_dict:
            scenario['domain'] = scenario['domain'].value
            scenario['complexity'] = scenario['complexity'].value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(scenarios_dict, f, indent=2, ensure_ascii=False)
    
    def generate_structured_dataset(self, total_count: int) -> List[CodeScenario]:
        """Generate a well-structured dataset covering all domains and complexities"""
        
        # Ensure coverage of all domains
        scenarios_per_domain = total_count // len(Domain)
        remaining = total_count % len(Domain)
        
        all_scenarios = []
        
        for i, domain in enumerate(Domain):
            domain_count = scenarios_per_domain + (1 if i < remaining else 0)
            
            # Distribute complexities within domain
            complexity_counts = {
                Complexity.BEGINNER: int(domain_count * 0.3),
                Complexity.INTERMEDIATE: int(domain_count * 0.4),
                Complexity.ADVANCED: int(domain_count * 0.2),
                Complexity.EXPERT: int(domain_count * 0.1)
            }
            
            # Adjust for rounding
            total_assigned = sum(complexity_counts.values())
            if total_assigned < domain_count:
                complexity_counts[Complexity.INTERMEDIATE] += domain_count - total_assigned
            
            # Generate scenarios for this domain
            for complexity, count in complexity_counts.items():
                for _ in range(count):
                    language = random.choice(["python", "javascript", "java", "cpp", "go", "typescript"])
                    context_type = random.choice(["startup", "enterprise", "open_source", "research", "education"])
                    
                    scenario = self.generate_scenario(domain, complexity, language, context_type)
                    all_scenarios.append(scenario)
        
        return all_scenarios


# Example usage and demonstration
def main():
    """Demonstrate the synthetic code generator"""
    generator = SyntheticCodeGenerator()
    
    # Generate single scenarios for each domain
    print("=== Sample Scenarios by Domain ===\n")
    
    for domain in list(Domain)[:3]:  # Show first 3 domains as examples
        scenario = generator.generate_scenario(
            domain=domain,
            complexity=Complexity.INTERMEDIATE,
            language="python",
            context_type="startup"
        )
        
        print(f"**Domain: {domain.name}**")
        print(f"Question: {scenario.question}")
        print(f"Context: {scenario.realistic_context}")
        print(f"Tags: {', '.join(scenario.tags)}")
        print("-" * 80)
        print()
    
    # Generate a batch with custom distribution
    print("=== Generating Batch Dataset ===\n")
    
    # Custom distribution favoring certain domains
    custom_distribution = {
        Domain.DATA_ANALYSIS: 0.2,
        Domain.DESKTOP_WEB_DEV: 0.2,
        Domain.SOFTWARE_ENGINEERING: 0.15,
        Domain.MACHINE_LEARNING: 0.15,
        Domain.BASIC_PROGRAMMING: 0.1,
        Domain.ADVANCED_PROGRAMMING: 0.1,
        Domain.DATABASE: 0.08,
        Domain.OTHERS: 0.02
    }
    
    # Custom complexity distribution
    complexity_dist = {
        Complexity.BEGINNER: 0.2,
        Complexity.INTERMEDIATE: 0.5,
        Complexity.ADVANCED: 0.25,
        Complexity.EXPERT: 0.05
    }
    
    batch_scenarios = generator.generate_batch(
        count=50,
        distribution=custom_distribution,
        complexity_distribution=complexity_dist,
        languages=["python", "javascript", "typescript", "java"]
    )
    
    print(f"Generated {len(batch_scenarios)} scenarios")
    
    # Analyze the generated batch
    domain_counts = {}
    complexity_counts = {}
    language_counts = {}
    
    for scenario in batch_scenarios:
        domain_counts[scenario.domain] = domain_counts.get(scenario.domain, 0) + 1
        complexity_counts[scenario.complexity] = complexity_counts.get(scenario.complexity, 0) + 1
        language_counts[scenario.language] = language_counts.get(scenario.language, 0) + 1
    
    print("\n**Distribution Analysis:**")
    print("Domains:", {d.name: c for d, c in domain_counts.items()})
    print("Complexities:", {c.value: cnt for c, cnt in complexity_counts.items()})
    print("Languages:", language_counts)
    
    # Generate structured dataset
    print("\n=== Generating Structured Dataset ===\n")
    structured_scenarios = generator.generate_structured_dataset(100)
    print(f"Generated {len(structured_scenarios)} structured scenarios")
    
    # Export to JSON
    generator.export_scenarios(structured_scenarios, "synthetic_code_dataset.json")
    print("Dataset exported to 'synthetic_code_dataset.json'")
    
    return generator, structured_scenarios


# Enhanced template system for more realistic scenarios
class AdvancedTemplateSystem:
    """Advanced template system for generating more sophisticated scenarios"""
    
    def __init__(self):
        self.scenario_chains = self._initialize_scenario_chains()
        self.real_world_contexts = self._initialize_real_world_contexts()
        self.technical_constraints = self._initialize_technical_constraints()
    
    def _initialize_scenario_chains(self):
        """Initialize multi-step scenario chains that build upon each other"""
        return {
            "web_app_development": [
                "I'm starting a new web application project. What's the best architecture approach?",
                "Now I need to implement user authentication for my web app. How should I handle this securely?",
                "My web app is getting popular and I'm seeing performance issues. How can I optimize it?",
                "I need to add real-time features to my web app. What's the best approach for WebSocket implementation?",
                "The web app needs to scale to handle 100k+ users. How should I redesign the architecture?"
            ],
            "data_pipeline_development": [
                "I need to build a data pipeline to process customer data from multiple sources. Where do I start?",
                "My data pipeline is working but it's slow. How can I optimize it for better performance?",
                "I need to add real-time processing capabilities to my data pipeline. What's the best approach?",
                "How can I make my data pipeline fault-tolerant and handle failures gracefully?",
                "I need to implement data quality monitoring in my pipeline. What should I track?"
            ],
            "microservices_development": [
                "I'm breaking down a monolith into microservices. What's the best strategy?",
                "How should I handle communication between my microservices?",
                "I need to implement distributed transaction management across microservices. How?",
                "My microservices architecture is becoming complex. How can I manage service discovery?",
                "How do I implement proper monitoring and observability for my microservices?"
            ]
        }
    
    def _initialize_real_world_contexts(self):
        """Initialize realistic business and technical contexts"""
        return {
            "fintech": {
                "constraints": ["PCI compliance required", "99.99% uptime needed", "audit trails mandatory"],
                "scenarios": ["payment processing", "fraud detection", "regulatory reporting"],
                "technologies": ["blockchain", "encrypted databases", "real-time analytics"]
            },
            "healthcare": {
                "constraints": ["HIPAA compliance", "patient data privacy", "medical device integration"],
                "scenarios": ["patient record management", "medical imaging", "telemedicine platforms"],
                "technologies": ["HL7 FHIR", "medical device APIs", "secure messaging"]
            },
            "ecommerce": {
                "constraints": ["high traffic during sales", "inventory synchronization", "payment security"],
                "scenarios": ["product catalog management", "order processing", "recommendation systems"],
                "technologies": ["CDN optimization", "real-time inventory", "ML recommendations"]
            },
            "gaming": {
                "constraints": ["low latency required", "anti-cheat systems", "scalable multiplayer"],
                "scenarios": ["game state synchronization", "matchmaking systems", "in-game economies"],
                "technologies": ["real-time networking", "game engines", "analytics platforms"]
            }
        }
    
    def _initialize_technical_constraints(self):
        """Initialize realistic technical constraints and requirements"""
        return {
            "performance": ["sub-100ms response time", "handle 10k concurrent users", "process 1M records/hour"],
            "scalability": ["horizontal scaling", "auto-scaling", "global distribution"],
            "security": ["zero-trust architecture", "end-to-end encryption", "secure by design"],
            "reliability": ["99.9% uptime", "disaster recovery", "graceful degradation"],
            "compliance": ["GDPR compliance", "SOC2 certification", "audit logging"]
        }
    
    def generate_contextual_scenario(self, domain: Domain, industry: str = None) -> Dict:
        """Generate scenarios with rich industry context"""
        if industry and industry in self.real_world_contexts:
            context = self.real_world_contexts[industry]
            constraint = random.choice(context["constraints"])
            scenario_type = random.choice(context["scenarios"])
            tech_stack = random.choice(context["technologies"])
            
            return {
                "industry": industry,
                "constraint": constraint,
                "scenario_type": scenario_type,
                "technology": tech_stack,
                "domain": domain
            }
        
        return {"domain": domain}


# Quality assessment system
class ScenarioQualityAssessor:
    """Assess the quality and realism of generated scenarios"""
    
    def __init__(self):
        self.quality_metrics = {
            "realism": 0.0,
            "technical_depth": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "practical_value": 0.0
        }
    
    def assess_scenario(self, scenario: CodeScenario) -> Dict[str, float]:
        """Assess the quality of a generated scenario"""
        scores = {}
        
        # Realism assessment
        scores["realism"] = self._assess_realism(scenario)
        
        # Technical depth assessment
        scores["technical_depth"] = self._assess_technical_depth(scenario)
        
        # Completeness assessment
        scores["completeness"] = self._assess_completeness(scenario)
        
        # Clarity assessment  
        scores["clarity"] = self._assess_clarity(scenario)
        
        # Practical value assessment
        scores["practical_value"] = self._assess_practical_value(scenario)
        
        # Overall score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _assess_realism(self, scenario: CodeScenario) -> float:
        """Assess how realistic the scenario is"""
        realism_indicators = [
            "production" in scenario.question.lower(),
            "scale" in scenario.question.lower(),
            "performance" in scenario.question.lower(),
            "security" in scenario.question.lower(),
            len(scenario.context) > 50,
            len(scenario.tags) >= 3
        ]
        return sum(realism_indicators) / len(realism_indicators)
    
    def _assess_technical_depth(self, scenario: CodeScenario) -> float:
        """Assess the technical depth of the scenario"""
        depth_indicators = [
            scenario.complexity in [Complexity.ADVANCED, Complexity.EXPERT],
            "implement" in scenario.question.lower(),
            "optimize" in scenario.question.lower(),
            "architecture" in scenario.question.lower(),
            len(scenario.answer) > 500
        ]
        return sum(depth_indicators) / len(depth_indicators)
    
    def _assess_completeness(self, scenario: CodeScenario) -> float:
        """Assess how complete the scenario is"""
        completeness_indicators = [
            len(scenario.question) > 50,
            len(scenario.answer) > 200,
            len(scenario.context) > 30,
            "best practices" in scenario.answer.lower(),
            "```" in scenario.answer  # Has code examples
        ]
        return sum(completeness_indicators) / len(completeness_indicators)
    
    def _assess_clarity(self, scenario: CodeScenario) -> float:
        """Assess how clear and well-structured the scenario is"""
        clarity_indicators = [
            "?" in scenario.question,
            scenario.question.count(" ") >= 10,  # Reasonable length
            scenario.answer.count("\n") >= 5,    # Well-structured
            "Example" in scenario.answer or "```" in scenario.answer
        ]
        return sum(clarity_indicators) / len(clarity_indicators)
    
    def _assess_practical_value(self, scenario: CodeScenario) -> float:
        """Assess the practical value of the scenario"""
        practical_indicators = [
            any(word in scenario.question.lower() for word in ["how", "implement", "build", "create"]),
            any(word in scenario.answer.lower() for word in ["solution", "approach", "method"]),
            scenario.domain != Domain.OTHERS,
            len(scenario.realistic_context) > 20
        ]
        return sum(practical_indicators) / len(practical_indicators)


# Integration with quality assessment
def generate_high_quality_dataset(count: int) -> List[CodeScenario]:
    """Generate a high-quality dataset with quality filtering"""
    generator = SyntheticCodeGenerator()
    assessor = ScenarioQualityAssessor()
    
    high_quality_scenarios = []
    attempts = 0
    max_attempts = count * 3  # Allow some generation overhead
    
    while len(high_quality_scenarios) < count and attempts < max_attempts:
        # Generate scenario
        domain = random.choice(list(Domain))
        complexity = random.choices(
            [Complexity.BEGINNER, Complexity.INTERMEDIATE, Complexity.ADVANCED, Complexity.EXPERT],
            weights=[0.2, 0.4, 0.3, 0.1]
        )[0]
        language = random.choice(["python", "javascript", "java", "typescript", "go"])
        context_type = random.choice(["startup", "enterprise", "open_source", "research"])
        
        scenario = generator.generate_scenario(domain, complexity, language, context_type)
        
        # Assess quality
        quality_scores = assessor.assess_scenario(scenario)
        
        # Filter based on quality threshold
        if quality_scores["overall"] >= 0.6:  # Adjustable threshold
            high_quality_scenarios.append(scenario)
        
        attempts += 1
    
    print(f"Generated {len(high_quality_scenarios)} high-quality scenarios in {attempts} attempts")
    return high_quality_scenarios


if __name__ == "__main__":
    # main()
    # Initialize the generator
    generator = SyntheticCodeGenerator()

    # Generate single scenario
    scenario = generator.generate_scenario(
        domain=Domain.DATA_ANALYSIS,
        complexity=Complexity.INTERMEDIATE,
        language="python",
        context_type="startup"
    )

    # Generate batch with custom distribution
    scenarios = generator.generate_batch(
        count=1000,
        distribution={Domain.DATA_ANALYSIS: 0.3, Domain.MACHINE_LEARNING: 0.2},
        languages=["python", "javascript", "java"]
    )

    # Generate high-quality filtered dataset
    quality_scenarios = generate_high_quality_dataset(500)