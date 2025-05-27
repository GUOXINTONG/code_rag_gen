"""
Knowledge graph construction and management for programming concepts.
"""

import random
import json
import networkx as nx
from typing import Dict, Set, List, Any
from collections import defaultdict

from data_structures import CodeConcept


class KnowledgeGraph:
    """Constructs and manages a knowledge graph of programming concepts."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.concepts: Dict[str, CodeConcept] = {}
        self.concept_categories: Dict[str, Set[str]] = defaultdict(set)
        self.educational_levels: Dict[str, Set[str]] = defaultdict(set)
        self.subject_areas: Dict[str, Set[str]] = defaultdict(set)
    
    def add_concept(self, concept: CodeConcept) -> None:
        """Add a concept to the knowledge graph."""
        self.concepts[concept.name] = concept
        self.concept_categories[concept.category].add(concept.name)
        self.educational_levels[concept.educational_level].add(concept.name)
        self.subject_areas.setdefault(concept.source_document, set()).add(concept.name)
        
        # Add node to graph with metadata
        self.graph.add_node(
            concept.name, 
            category=concept.category,
            complexity=concept.complexity_level,
            description=concept.description,
            educational_level=concept.educational_level,
            source_document=concept.source_document
        )
        
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
                                 diversity_weight: float = 0.7,
                                 prefer_connected: bool = True) -> List[CodeConcept]:
        """
        Sample a diverse combination of concepts for question generation.
        
        Args:
            num_concepts: Number of concepts to select
            diversity_weight: Weight for diversity vs randomness (0-1)
            prefer_connected: Whether to prefer concepts that are connected in the graph
        """
        if len(self.concepts) < num_concepts:
            return list(self.concepts.values())
        
        selected_concepts = []
        selected_categories = set()
        selected_complexity = set()
        selected_sources = set()
        
        available_concepts = list(self.concepts.keys())
        random.shuffle(available_concepts)
        
        # First, select a seed concept randomly
        if available_concepts:
            seed_concept_name = random.choice(available_concepts)
            seed_concept = self.concepts[seed_concept_name]
            selected_concepts.append(seed_concept)
            selected_categories.add(seed_concept.category)
            selected_complexity.add(seed_concept.complexity_level)
            selected_sources.add(seed_concept.source_document)
            available_concepts.remove(seed_concept_name)
        
        # Select remaining concepts with diversity and connectivity considerations
        for concept_name in available_concepts:
            if len(selected_concepts) >= num_concepts:
                break
                
            concept = self.concepts[concept_name]
            
            # Calculate diversity score
            diversity_score = 0
            if concept.category not in selected_categories:
                diversity_score += 1
            if concept.complexity_level not in selected_complexity:
                diversity_score += 1
            if concept.source_document not in selected_sources:
                diversity_score += 0.5
            
            # Calculate connectivity score if prefer_connected is True
            connectivity_score = 0
            if prefer_connected and selected_concepts:
                for selected in selected_concepts:
                    if self.graph.has_edge(concept_name, selected.name):
                        connectivity_score += 1
                connectivity_score = min(connectivity_score / len(selected_concepts), 1.0)
            
            # Combined acceptance probability
            base_prob = diversity_weight * (diversity_score / 2.5) + (1 - diversity_weight) * random.random()
            if prefer_connected:
                base_prob += 0.2 * connectivity_score
            
            # Accept based on probability threshold
            if base_prob > 0.3 or len(selected_concepts) < 2:
                selected_concepts.append(concept)
                selected_categories.add(concept.category)
                selected_complexity.add(concept.complexity_level)
                selected_sources.add(concept.source_document)
        
        # Fill remaining slots randomly if needed
        while len(selected_concepts) < num_concepts and len(selected_concepts) < len(self.concepts):
            remaining = [c for name, c in self.concepts.items() 
                        if c not in selected_concepts]
            if remaining:
                selected_concepts.append(random.choice(remaining))
        
        return selected_concepts
    
    def get_concepts_by_category(self, category: str) -> List[CodeConcept]:
        """Get all concepts belonging to a specific category."""
        return [self.concepts[name] for name in self.concept_categories.get(category, [])]
    
    def get_concepts_by_educational_level(self, level: str) -> List[CodeConcept]:
        """Get all concepts belonging to a specific educational level."""
        return [self.concepts[name] for name in self.educational_levels.get(level, [])]
    
    def find_concept_clusters(self, min_cluster_size: int = 3) -> List[Set[str]]:
        """Find clusters of related concepts in the graph."""
        try:
            # Use community detection to find clusters
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(self.graph)
            return [cluster for cluster in communities if len(cluster) >= min_cluster_size]
        except ImportError:
            # Fallback: use connected components
            components = list(nx.connected_components(self.graph))
            return [comp for comp in components if len(comp) >= min_cluster_size]
    
    def get_concept_path(self, concept1: str, concept2: str) -> List[str]:
        """Get the shortest path between two concepts."""
        try:
            return nx.shortest_path(self.graph, concept1, concept2)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        return {
            "total_concepts": len(self.concepts),
            "categories": {cat: len(concepts) for cat, concepts in self.concept_categories.items()},
            "educational_levels": {level: len(concepts) for level, concepts in self.educational_levels.items()},
            "graph_density": nx.density(self.graph) if self.graph.nodes() else 0,
            "connected_components": nx.number_connected_components(self.graph),
            "average_clustering": nx.average_clustering(self.graph) if self.graph.nodes() else 0,
            "total_edges": self.graph.number_of_edges(),
            "most_connected_concepts": self._get_most_connected_concepts(5)
        }
    
    def _get_most_connected_concepts(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get the most connected concepts in the graph."""
        if not self.graph.nodes():
            return []
        
        degree_centrality = nx.degree_centrality(self.graph)
        top_concepts = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {
                "concept": concept_name,
                "degree": self.graph.degree(concept_name),
                "centrality": centrality,
                "category": self.concepts[concept_name].category
            }
            for concept_name, centrality in top_concepts
        ]
    
    def export_graph(self, filepath: str, format: str = "json") -> None:
        """Export the knowledge graph in various formats."""
        if format == "json":
            self._export_json(filepath)
        elif format == "gexf":
            nx.write_gexf(self.graph, filepath)
        elif format == "graphml":
            nx.write_graphml(self.graph, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, filepath: str) -> None:
        """Export knowledge graph as JSON."""
        graph_data = {
            "concepts": {
                name: {
                    "category": concept.category,
                    "description": concept.description,
                    "complexity_level": concept.complexity_level,
                    "educational_level": concept.educational_level,
                    "related_concepts": concept.related_concepts,
                    "source_document": concept.source_document,
                    "code_examples": concept.code_examples
                }
                for name, concept in self.concepts.items()
            },
            "edges": list(self.graph.edges()),
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    def load_graph(self, filepath: str) -> None:
        """Load knowledge graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        self.concepts.clear()
        self.concept_categories.clear()
        self.educational_levels.clear()
        self.subject_areas.clear()
        
        # Load concepts
        for name, concept_data in graph_data["concepts"].items():
            concept = CodeConcept(
                name=name,
                category=concept_data["category"],
                description=concept_data["description"],
                complexity_level=concept_data["complexity_level"],
                educational_level=concept_data["educational_level"],
                related_concepts=concept_data["related_concepts"],
                code_examples=concept_data["code_examples"],
                source_document=concept_data["source_document"]
            )
            self.add_concept(concept)