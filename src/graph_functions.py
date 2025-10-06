# ============================================================================
# NETWORK GRAPH HELPER FUNCTIONS
# ============================================================================
# graph_functions.py
# By Alistair Schillert 
# HELPER FUNCTIONS FOR NETWORK GRAPH
import pickle 
import networkx as nx
from pyvis.network import Network
import streamlit as st
# ============================================================================
# Function 1: Load Graph: This is to specifically a test function to upload a pickle file
# ============================================================================
def load_graph_from_pickle(file):
    """Load NetworkX graph from pickle file"""
    try:
        file.seek(0)
        G = pickle.load(file)
        if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            st.error("Uploaded file is not a NetworkX graph")
            return None
        return G
    except EOFError:
        st.error("Error: Pickle file appears to be incomplete or corrupted")
        return None
    except pickle.UnpicklingError as e:
        st.error(f"Error unpickling file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading pickle file: {str(e)}")
        return None
# ============================================================================
#Function 2: Load Citation Graph, which autoloads the legal citation graph specifically with the intention of loading a given file. 
# ============================================================================
def load_citation_graph():
    """Auto-load citation_graph.pkl if it exists"""
    try:
        with open("citation_graph.pkl", "rb") as f:
            G = pickle.load(f)
            if isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return G
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"Could not auto-load citation_graph.pkl: {str(e)}")
    return None
# ============================================================================
#Functiion 3: Create PyVis Graph:
#This is to specificlaly create Pyvis graph, based upon performance optimsiation settings
#This allows for our given file to be loaded in, with pre-selected settings with each of the settings). 
def create_pyvis_graph(G, search_term="", node_size_multiplier=2, edge_width=1, arrow_size=8, node_spacing=800, edge_length=800, layout_algorithm="Force Atlas 2 (Sparse)", use_cached_layout=True, render_quality="Balanced", max_edges=10000, edge_opacity=0.3, overlap_strength=3.5, enable_highlighting=True, highlight_colour_in="#00FF00", highlight_colour_out="#FF00FF"):
    """Convert NetworkX graph to PyVis with performance optimisations"""
    edges_to_render = list(G.edges())
    total_edges = len(edges_to_render)
    
    if max_edges > 0 and total_edges > max_edges:
        st.warning(f"⚠️ Graph has {total_edges} edges. Displaying top {max_edges} edges by node importance.")
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            node_importance = {node: G.in_degree(node) + G.out_degree(node) for node in G.nodes()}
        else:
            node_importance = dict(G.degree())
        edge_scores = [(edge, node_importance.get(edge[0], 0) + node_importance.get(edge[1], 0)) 
                      for edge in edges_to_render]
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        edges_to_render = [edge for edge, score in edge_scores[:max_edges]]
    
    if use_cached_layout:
        try:
            if layout_algorithm == "Shell (Concentric)":
                if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                    degrees = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
                else:
                    degrees = dict(G.degree())
                max_deg = max(degrees.values()) if degrees else 1
                shells = [[] for _ in range(4)]
                for node, deg in degrees.items():
                    if deg == 0:
                        shells[3].append(node)
                    elif deg < max_deg * 0.1:
                        shells[2].append(node)
                    elif deg < max_deg * 0.5:
                        shells[1].append(node)
                    else:
                        shells[0].append(node)
                shells = [s for s in shells if s]
                pos = nx.shell_layout(G, nlist=shells, scale=node_spacing * 2)
            else:
                pos = nx.spring_layout(G, k=edge_length/40, iterations=150, seed=42, scale=node_spacing * 2.5)
            for node in pos:
                pos[node] = (pos[node][0] * 1.8, pos[node][1] * 1.8)
        except Exception as e:
            st.warning(f"⚠️ Layout computation issue: {str(e)[:100]}")
            pos = None
    else:
        pos = None
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='black',
                  directed=isinstance(G, (nx.DiGraph, nx.MultiDiGraph)))
    net.toggle_physics(True)
    
    if render_quality == "Fast":
        stabilization_iterations = 100
        smooth_type = "discrete"
        hide_edges_on_drag = True
        hide_edges_on_zoom = True
    elif render_quality == "High Quality":
        stabilization_iterations = 2000
        smooth_type = "continuous"
        hide_edges_on_drag = False
        hide_edges_on_zoom = False
    else:
        stabilization_iterations = 500
        smooth_type = "discrete"
        hide_edges_on_drag = True
        hide_edges_on_zoom = True
    
    options = """
    {
        "nodes": {
            "font": {"size": 11, "face": "arial", "strokeWidth": 0, "vadjust": 0},
            "scaling": {"min": 15, "max": 80},
            "shape": "dot",
            "borderWidth": 2
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": """ + str(arrow_size / 10) + """}},
            "smooth": {"type": \"""" + smooth_type + """\", "roundness": 0.2},
            "length": """ + str(edge_length) + """
        },
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": """ + str(stabilization_iterations) + """},
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -1500,
                "centralGravity": 0.005,
                "springLength": """ + str(edge_length) + """,
                "damping": 0.96,
                "avoidOverlap": 1
            },
            "timestep": 0.5
        },
        "interaction": {
            "hover": true,
            "hideEdgesOnDrag": """ + str(hide_edges_on_drag).lower() + """,
            "hideEdgesOnZoom": """ + str(hide_edges_on_zoom).lower() + """,
            "navigationButtons": true
        }
    }
    """
    net.set_options(options)
    
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        degrees = dict(G.in_degree())
    else:
        degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        label = G.nodes[node].get('label', str(node))
        if len(label) > 50:
            label = label[:50] + "..."
        degree = degrees[node]
        import math
        if max_degree > 0:
            normalized_size = math.log(1 + degree) / math.log(1 + max_degree)
            size = (15 + normalized_size * 45) * node_size_multiplier
        else:
            size = 20 * node_size_multiplier
        font_size = max(9, min(11, int(size / 4)))
        
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            title = f"Node: {label}\nIn-degree: {G.in_degree(node)}\nOut-degree: {G.out_degree(node)}"
        else:
            title = f"Node: {label}\nDegree: {G.degree(node)}"
        
        node_matches_search = False
        if search_term:
            search_lower = search_term.lower()
            node_matches_search = search_lower in label.lower() or search_lower in str(node).lower()
        
        if node_matches_search:
            color = '#FFD700'
            border_color = '#FF8C00'
            border_width = 5
            font_size = min(14, font_size + 2)
        else:
            if max_degree > 0:
                intensity = int(151 + (degree / max_degree) * 100)
                color = f'#{intensity:02x}{194:02x}{252:02x}'
            else:
                color = '#97C2FC'
            border_color = '#2B7CE9'
            border_width = 2
        
        node_params = {
            'label': label,
            'title': title,
            'size': size,
            'color': {'background': color, 'border': border_color},
            'borderWidth': border_width,
            'font': {'size': font_size, 'bold': node_matches_search}
        }
        
        if pos is not None and node in pos:
            node_params['x'] = pos[node][0]
            node_params['y'] = pos[node][1]
            node_params['physics'] = False
        
        net.add_node(node, **node_params)
    
    highlighted_edges = set()
    if search_term:
        matching_nodes = set()
        search_lower = search_term.lower()
        for node in G.nodes():
            node_label = G.nodes[node].get('label', str(node))
            if search_lower in node_label.lower() or search_lower in str(node).lower():
                matching_nodes.add(node)
        for edge in edges_to_render:
            if edge[0] in matching_nodes or edge[1] in matching_nodes:
                highlighted_edges.add(edge)
    
    for edge in edges_to_render:
        if edge in highlighted_edges:
            net.add_edge(edge[0], edge[1], width=edge_width * 3, color={'color': '#FF6600', 'opacity': 0.9})
        else:
            net.add_edge(edge[0], edge[1], width=edge_width, color={'opacity': edge_opacity})
    
    return net, total_edges, len(edges_to_render)