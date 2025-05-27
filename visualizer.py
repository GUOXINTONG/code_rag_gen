"""
Knowledge Graph Visualizer for the Level-3 Code Question Generator.
Provides interactive visualizations of programming concept relationships.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import colorsys

from knowledge_graph import KnowledgeGraph
from data_structures import CodeConcept


class KnowledgeGraphVisualizer:
    """
    Interactive visualizer for programming concept knowledge graphs.
    Provides multiple visualization types and analysis tools.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.category_colors = self._generate_category_colors()
        self.complexity_colors = {
            'beginner': '#4CAF50',      # Green
            'intermediate': '#FF9800',   # Orange  
            'advanced': '#F44336',       # Red
            'expert': '#9C27B0'          # Purple
        }
        
    def _generate_category_colors(self) -> Dict[str, str]:
        """Generate distinct colors for each concept category."""
        categories = list(self.kg.concept_categories.keys())
        colors = {}
        
        # Generate evenly spaced hues
        for i, category in enumerate(categories):
            hue = i / len(categories)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors[category] = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
        
        return colors
    
    def create_interactive_network(self, 
                                 layout: str = 'spring',
                                 color_by: str = 'category',
                                 size_by: str = 'degree',
                                 show_labels: bool = True,
                                 filter_category: Optional[str] = None,
                                 min_connections: int = 0) -> go.Figure:
        """
        Create an interactive network visualization using Plotly.
        
        Args:
            layout: Network layout algorithm ('spring', 'circular', 'random', 'shell')
            color_by: How to color nodes ('category', 'complexity', 'source_document')
            size_by: How to size nodes ('degree', 'uniform', 'complexity')
            show_labels: Whether to show node labels
            filter_category: Show only nodes from this category (None for all)
            min_connections: Minimum number of connections for nodes to be shown
        """
        # Filter graph if needed
        graph = self.kg.graph.copy()
        if filter_category:
            category_nodes = self.kg.concept_categories.get(filter_category, set())
            graph = graph.subgraph(category_nodes)
        
        if min_connections > 0:
            nodes_to_keep = [node for node, degree in graph.degree() if degree >= min_connections]
            graph = graph.subgraph(nodes_to_keep)
        
        if not graph.nodes():
            return go.Figure().add_annotation(text="No nodes to display with current filters", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'random':
            pos = nx.random_layout(graph)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Prepare node data
        node_trace = self._create_node_trace(graph, pos, color_by, size_by)
        edge_trace = self._create_edge_trace(graph, pos)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title=f"Programming Concepts Knowledge Graph ({len(graph.nodes())} nodes, {len(graph.edges())} edges)",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details. Drag to pan, scroll to zoom.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#888', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_node_trace(self, graph, pos, color_by, size_by):
        """Create node trace for Plotly visualization."""
        x_nodes = [pos[node][0] for node in graph.nodes()]
        y_nodes = [pos[node][1] for node in graph.nodes()]
        
        # Determine node colors
        node_colors = []
        colorscale = None
        
        for node in graph.nodes():
            concept = self.kg.concepts[node]
            if color_by == 'category':
                node_colors.append(self.category_colors.get(concept.category, '#888'))
            elif color_by == 'complexity':
                node_colors.append(self.complexity_colors.get(concept.complexity_level, '#888'))
            elif color_by == 'source_document':
                # Hash source document to color
                hash_val = hash(concept.source_document) % 360
                rgb = colorsys.hsv_to_rgb(hash_val/360, 0.7, 0.9)
                node_colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
        
        # Determine node sizes
        if size_by == 'degree':
            degrees = [graph.degree(node) for node in graph.nodes()]
            max_degree = max(degrees) if degrees else 1
            node_sizes = [10 + (degree / max_degree) * 30 for degree in degrees]
        elif size_by == 'complexity':
            complexity_map = {'beginner': 10, 'intermediate': 20, 'advanced': 30, 'expert': 40}
            node_sizes = [complexity_map.get(self.kg.concepts[node].complexity_level, 15) 
                         for node in graph.nodes()]
        else:  # uniform
            node_sizes = [15] * len(graph.nodes())
        
        # Create hover text
        hover_texts = []
        for node in graph.nodes():
            concept = self.kg.concepts[node]
            connections = len(list(graph.neighbors(node)))
            
            hover_text = f"<b>{concept.name}</b><br>"
            hover_text += f"Category: {concept.category}<br>"
            hover_text += f"Complexity: {concept.complexity_level}<br>"
            hover_text += f"Source: {concept.source_document}<br>"
            hover_text += f"Connections: {connections}<br>"
            hover_text += f"Description: {concept.description[:100]}..."
            
            hover_texts.append(hover_text)
        
        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_texts,
            text=[node if len(node) < 15 else node[:12]+'...' for node in graph.nodes()],
            textposition="middle center",
            textfont=dict(size=8),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='#888'),
                opacity=0.8
            )
        )
        
        return node_trace
    
    def _create_edge_trace(self, graph, pos):
        """Create edge trace for Plotly visualization."""
        x_edges = []
        y_edges = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            x_edges.extend([x0, x1, None])
            y_edges.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=x_edges, y=y_edges,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            opacity=0.5
        )
        
        return edge_trace
    
    def create_category_analysis_dashboard(self) -> go.Figure:
        """Create a comprehensive dashboard analyzing concept categories."""
        # Prepare data
        category_stats = {}
        for category, concepts in self.kg.concept_categories.items():
            concepts_list = [self.kg.concepts[name] for name in concepts]
            category_stats[category] = {
                'count': len(concepts),
                'avg_connections': np.mean([self.kg.graph.degree(name) for name in concepts]),
                'complexity_dist': Counter(c.complexity_level for c in concepts_list),
                'source_dist': Counter(c.source_document for c in concepts_list)
            }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Concepts per Category', 'Average Connections per Category',
                          'Complexity Distribution', 'Category Network Centrality'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        categories = list(category_stats.keys())
        
        # Concepts per category
        fig.add_trace(
            go.Bar(x=categories, 
                   y=[category_stats[cat]['count'] for cat in categories],
                   name='Concept Count',
                   marker_color=[self.category_colors.get(cat, '#888') for cat in categories]),
            row=1, col=1
        )
        
        # Average connections per category
        fig.add_trace(
            go.Bar(x=categories,
                   y=[category_stats[cat]['avg_connections'] for cat in categories],
                   name='Avg Connections',
                   marker_color=[self.category_colors.get(cat, '#888') for cat in categories]),
            row=1, col=2
        )
        
        # Complexity distribution (stacked bar)
        complexity_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        for i, complexity in enumerate(complexity_levels):
            fig.add_trace(
                go.Bar(x=categories,
                       y=[category_stats[cat]['complexity_dist'].get(complexity, 0) for cat in categories],
                       name=complexity.title(),
                       marker_color=self.complexity_colors.get(complexity, '#888')),
                row=2, col=1
            )
        
        # Category centrality (using betweenness centrality)
        try:
            centrality = nx.betweenness_centrality(self.kg.graph)
            category_centrality = defaultdict(list)
            for node, cent in centrality.items():
                category = self.kg.concepts[node].category
                category_centrality[category].append(cent)
            
            avg_centrality = {cat: np.mean(cents) for cat, cents in category_centrality.items()}
            
            fig.add_trace(
                go.Scatter(x=list(avg_centrality.keys()),
                          y=list(avg_centrality.values()),
                          mode='markers',
                          marker=dict(size=15, 
                                    color=[self.category_colors.get(cat, '#888') 
                                          for cat in avg_centrality.keys()]),
                          name='Avg Centrality'),
                row=2, col=2
            )
        except:
            # Fallback if centrality calculation fails
            fig.add_annotation(text="Centrality calculation unavailable", 
                             xref="x4", yref="y4", x=0.5, y=0.5, row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Knowledge Graph Category Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_concept_similarity_heatmap(self, top_n: int = 20) -> go.Figure:
        """Create a heatmap showing concept similarities based on shared connections."""
        # Get top N most connected concepts
        degrees = dict(self.kg.graph.degree())
        top_concepts = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        concept_names = [name for name, _ in top_concepts]
        
        # Calculate similarity matrix (Jaccard similarity of neighbors)
        similarity_matrix = np.zeros((len(concept_names), len(concept_names)))
        
        for i, concept1 in enumerate(concept_names):
            for j, concept2 in enumerate(concept_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    neighbors1 = set(self.kg.graph.neighbors(concept1))
                    neighbors2 = set(self.kg.graph.neighbors(concept2))
                    
                    if neighbors1 or neighbors2:
                        intersection = len(neighbors1.intersection(neighbors2))
                        union = len(neighbors1.union(neighbors2))
                        similarity_matrix[i][j] = intersection / union if union > 0 else 0
                    else:
                        similarity_matrix[i][j] = 0
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=concept_names,
            y=concept_names,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='%{x} - %{y}<br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Concept Similarity Heatmap (Top {top_n} Most Connected Concepts)',
            xaxis_title='Concepts',
            yaxis_title='Concepts',
            width=800,
            height=800
        )
        
        return fig
    
    def create_learning_path_visualization(self, 
                                         start_concept: str, 
                                         target_concepts: List[str]) -> go.Figure:
        """Visualize learning paths from a starting concept to target concepts."""
        if start_concept not in self.kg.concepts:
            return go.Figure().add_annotation(text=f"Concept '{start_concept}' not found", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Find paths to all target concepts
        paths = {}
        for target in target_concepts:
            if target in self.kg.concepts:
                try:
                    path = nx.shortest_path(self.kg.graph, start_concept, target)
                    paths[target] = path
                except nx.NetworkXNoPath:
                    continue
        
        if not paths:
            return go.Figure().add_annotation(text="No paths found to target concepts", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Create subgraph with all path nodes
        all_path_nodes = set([start_concept])
        for path in paths.values():
            all_path_nodes.update(path)
        
        subgraph = self.kg.graph.subgraph(all_path_nodes)
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Create visualization
        edge_trace = self._create_edge_trace(subgraph, pos)
        
        # Color nodes based on their role
        node_colors = []
        node_sizes = []
        hover_texts = []
        
        for node in subgraph.nodes():
            concept = self.kg.concepts[node]
            
            if node == start_concept:
                color = 'green'
                size = 25
                role = 'Start'
            elif node in target_concepts:
                color = 'red' 
                size = 25
                role = 'Target'
            else:
                color = self.category_colors.get(concept.category, '#888')
                size = 15
                role = 'Intermediate'
            
            node_colors.append(color)
            node_sizes.append(size)
            
            # Find which paths this node belongs to
            in_paths = [target for target, path in paths.items() if node in path]
            
            hover_text = f"<b>{concept.name}</b><br>"
            hover_text += f"Role: {role}<br>"
            hover_text += f"Category: {concept.category}<br>"
            hover_text += f"Complexity: {concept.complexity_level}<br>"
            if in_paths:
                hover_text += f"In paths to: {', '.join(in_paths)}<br>"
            
            hover_texts.append(hover_text)
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in subgraph.nodes()],
            y=[pos[node][1] for node in subgraph.nodes()],
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_texts,
            text=list(subgraph.nodes()),
            textposition="middle center",
            textfont=dict(size=10),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Add path information as annotations
        path_info = f"Learning Paths from '{start_concept}':<br>"
        for target, path in paths.items():
            path_info += f"→ {target}: {' → '.join(path)}<br>"
        
        fig.update_layout(
            title="Learning Path Visualization",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(text=path_info,
                     showarrow=False,
                     xref="paper", yref="paper",
                     x=0.02, y=0.98,
                     xanchor='left', yanchor='top',
                     font=dict(size=10),
                     bgcolor="rgba(255,255,255,0.8)",
                     bordercolor="black",
                     borderwidth=1)
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_complexity_progression_chart(self) -> go.Figure:
        """Show how concepts progress in complexity within each category."""
        fig = go.Figure()
        
        complexity_order = ['beginner', 'intermediate', 'advanced', 'expert']
        
        for category in self.kg.concept_categories.keys():
            category_concepts = [self.kg.concepts[name] 
                               for name in self.kg.concept_categories[category]]
            
            complexity_counts = Counter(c.complexity_level for c in category_concepts)
            
            fig.add_trace(go.Scatter(
                x=complexity_order,
                y=[complexity_counts.get(level, 0) for level in complexity_order],
                mode='lines+markers',
                name=category,
                line=dict(color=self.category_colors.get(category, '#888')),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Complexity Progression by Category',
            xaxis_title='Complexity Level',
            yaxis_title='Number of Concepts',
            hovermode='x unified'
        )
        
        return fig
    
    def create_static_network_plot(self, 
                                 layout: str = 'spring',
                                 figsize: Tuple[int, int] = (15, 12),
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create a static network visualization using matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(self.kg.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.kg.graph)
        else:
            pos = nx.spring_layout(self.kg.graph)
        
        # Draw edges
        nx.draw_networkx_edges(self.kg.graph, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Draw nodes colored by category
        for category in self.kg.concept_categories.keys():
            category_nodes = list(self.kg.concept_categories[category])
            if category_nodes:
                nx.draw_networkx_nodes(
                    self.kg.graph, pos, 
                    nodelist=category_nodes,
                    node_color=self.category_colors.get(category, '#888'),
                    node_size=[self.kg.graph.degree(node) * 20 + 50 for node in category_nodes],
                    alpha=0.7,
                    label=category,
                    ax=ax
                )
        
        # Add labels for highly connected nodes
        high_degree_nodes = [node for node, degree in self.kg.graph.degree() if degree >= 3]
        labels = {node: node if len(node) <= 12 else node[:10] + '...' 
                 for node in high_degree_nodes}
        
        nx.draw_networkx_labels(self.kg.graph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Programming Concepts Knowledge Graph\n'
                    f'{len(self.kg.concepts)} concepts, {self.kg.graph.number_of_edges()} connections',
                    fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate a comprehensive insights report about the knowledge graph."""
        stats = self.kg.get_stats()
        
        # Calculate additional metrics
        insights = {
            'basic_stats': stats,
            'category_analysis': {},
            'complexity_analysis': {},
            'connectivity_insights': {},
            'learning_path_recommendations': []
        }
        
        # Category analysis
        for category, concepts in self.kg.concept_categories.items():
            concept_objects = [self.kg.concepts[name] for name in concepts]
            connections = [self.kg.graph.degree(name) for name in concepts]
            
            insights['category_analysis'][category] = {
                'count': len(concepts),
                'avg_connections': np.mean(connections) if connections else 0,
                'max_connections': max(connections) if connections else 0,
                'complexity_distribution': dict(Counter(c.complexity_level for c in concept_objects)),
                'most_connected': max(concepts, key=lambda x: self.kg.graph.degree(x)) if concepts else None
            }
        
        # Complexity analysis
        complexity_concepts = defaultdict(list)
        for concept in self.kg.concepts.values():
            complexity_concepts[concept.complexity_level].append(concept.name)
        
        for complexity, concepts in complexity_concepts.items():
            connections = [self.kg.graph.degree(name) for name in concepts]
            insights['complexity_analysis'][complexity] = {
                'count': len(concepts),
                'avg_connections': np.mean(connections) if connections else 0,
                'categories': dict(Counter(self.kg.concepts[name].category for name in concepts))
            }
        
        # Connectivity insights
        if self.kg.graph.nodes():
            try:
                centrality = nx.betweenness_centrality(self.kg.graph)
                closeness = nx.closeness_centrality(self.kg.graph)
                
                insights['connectivity_insights'] = {
                    'most_central_concepts': sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5],
                    'most_accessible_concepts': sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5],
                    'bridge_concepts': [node for node, centrality in centrality.items() if centrality > 0.1],
                    'isolated_concepts': [node for node in self.kg.graph.nodes() if self.kg.graph.degree(node) == 0]
                }
            except:
                insights['connectivity_insights'] = {'error': 'Could not calculate centrality metrics'}
        
        # Learning path recommendations
        beginner_concepts = [name for name, concept in self.kg.concepts.items() 
                           if concept.complexity_level == 'beginner']
        advanced_concepts = [name for name, concept in self.kg.concepts.items() 
                           if concept.complexity_level in ['advanced', 'expert']]
        
        if beginner_concepts and advanced_concepts:
            sample_paths = []
            for start in beginner_concepts[:3]:
                for end in advanced_concepts[:3]:
                    try:
                        path = nx.shortest_path(self.kg.graph, start, end)
                        if len(path) > 2:  # Only interesting paths
                            sample_paths.append({
                                'from': start,
                                'to': end,
                                'path': path,
                                'length': len(path) - 1
                            })
                    except nx.NetworkXNoPath:
                        continue
            
            insights['learning_path_recommendations'] = sorted(sample_paths, key=lambda x: x['length'])[:5]
        
        return insights


# Example usage and integration
def create_visualization_demo(knowledge_graph: KnowledgeGraph):
    """Demo function showing how to use the visualizer."""
    visualizer = KnowledgeGraphVisualizer(knowledge_graph)
    
    # Create interactive network
    network_fig = visualizer.create_interactive_network(
        layout='spring',
        color_by='category',
        size_by='degree'
    )
    
    # Create analysis dashboard
    dashboard_fig = visualizer.create_category_analysis_dashboard()
    
    # Generate insights
    insights = visualizer.generate_insights_report()
    
    return {
        'network_visualization': network_fig,
        'analysis_dashboard': dashboard_fig,
        'insights': insights,
        'visualizer': visualizer
    }


if __name__ == "__main__":
    # Example usage with the Level3QuestionGenerator
    from level3_generator import Level3QuestionGenerator
    from data_structures import MockLLM
    
    # Create and populate knowledge graph
    mock_llm = MockLLM()
    generator = Level3QuestionGenerator(mock_llm)
    
    sample_documents = [
        "Data structures including arrays, trees, graphs, and hash tables...",
        "Algorithms covering sorting, searching, and optimization techniques...",
        "Object-oriented programming with classes, inheritance, and polymorphism..."
    ]
    
    generator.process_documents(sample_documents, ["ds", "algo", "oop"])
    
    # Create visualizations
    demo = create_visualization_demo(generator.knowledge_graph)
    
    # Display network (in Jupyter notebook or save to HTML)
    demo['network_visualization'].show()
    demo['analysis_dashboard'].show()
    
    # Print insights
    print("Knowledge Graph Insights:")
    for section, data in demo['insights'].items():
        print(f"\n{section.upper()}:")
        print(data)