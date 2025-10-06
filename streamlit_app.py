# streamlit_app.py - Integrated Version

import streamlit as st
import time
from pathlib import Path
import tempfile
import os
from difflib import SequenceMatcher
import networkx as nx
import pickle
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime

# Import summarization modules (lazy load to avoid conflicts)
from src.pdf_handler import extract_text_from_pdf
from src.summariser import initialise_summariser, summarise_text
from src import llm_handler
from prompts import PROMPT_OPTIONS

#Import Goo
import anthropic
import openai
from google import generativeai as genai

# PAGE CONFIG (MUST BE FIRST)
st.set_page_config(
    page_title="LegalEase NSW",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# LOAD CSS FILE
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file optional

load_css("static/style.css")

# HEADER
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    try:
        st.image("static/logo.jpg", width=150)
    except:
        pass
with col2:
    st.markdown("<h1>LegalEase NSW</h1>", unsafe_allow_html=True)
    st.markdown("<h3>AI-Powered Legal Document Summariser & Citation Network</h3>", unsafe_allow_html=True)
with col3:
    try:
        st.image("static/logo.jpg", width=150)
    except:
        pass

st.markdown("---")


# ============================================================================
# NETWORK GRAPH HELPER FUNCTIONS
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
    except Exception as e:
        st.error(f"Error loading pickle file: {str(e)}")
        return None

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

@st.cache_data(show_spinner=False)
def compute_graph_metrics(_G):
    """Cache degree calculations and node labels"""
    if isinstance(_G, (nx.DiGraph, nx.MultiDiGraph)):
        in_degrees = dict(_G.in_degree())
        out_degrees = dict(_G.out_degree())
    else:
        in_degrees = dict(_G.degree())
        out_degrees = None
    
    node_labels = {node: _G.nodes[node].get('label', str(node)) for node in _G.nodes()}
    max_degree = max(in_degrees.values()) if in_degrees else 1
    
    return in_degrees, out_degrees, node_labels, max_degree

@st.cache_data(show_spinner=False)
def filter_edges_by_importance(_G, max_edges):
    """Cache edge filtering computation"""
    edges_to_render = list(_G.edges())
    total_edges = len(edges_to_render)
    
    if max_edges > 0 and total_edges > max_edges:
        if isinstance(_G, (nx.DiGraph, nx.MultiDiGraph)):
            node_importance = {node: _G.in_degree(node) + _G.out_degree(node) for node in _G.nodes()}
        else:
            node_importance = dict(_G.degree())
        edge_scores = [(edge, node_importance.get(edge[0], 0) + node_importance.get(edge[1], 0)) 
                      for edge in edges_to_render]
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        edges_to_render = [edge for edge, score in edge_scores[:max_edges]]
    
    return edges_to_render, total_edges

@st.cache_data(show_spinner=False)
def compute_search_matches(_G, search_term, _node_labels):
    """Cache search result computation"""
    matching_nodes = set()
    if search_term:
        search_lower = search_term.lower()
        for node in _G.nodes():
            node_label = _node_labels[node]
            if search_lower in node_label.lower() or search_lower in str(node).lower():
                matching_nodes.add(node)
    return matching_nodes

def create_pyvis_graph(G, search_term="", node_size_multiplier=2, edge_width=1, arrow_size=8, 
                      edge_length=800, max_edges=10000, edge_opacity=0.3, 
                      highlight_colour_in="#00FF00", highlight_colour_out="#FF00FF"):
    """Convert NetworkX graph to PyVis with performance optimizations"""
    
    # Use cached edge filtering
    edges_to_render, total_edges = filter_edges_by_importance(G, max_edges)
    
    if max_edges > 0 and total_edges > max_edges:
        st.warning(f"⚠️ Graph has {total_edges} edges. Displaying top {len(edges_to_render)} edges by node importance.")
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='black',
                  directed=isinstance(G, (nx.DiGraph, nx.MultiDiGraph)))
    
    # Enable physics initially - will be disabled after 5 seconds via JavaScript
    net.toggle_physics(True)
    
    options = """
    {
        "nodes": {
            "font": {"size": 11, "face": "arial"},
            "shape": "dot",
            "borderWidth": 2
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": """ + str(arrow_size / 10) + """}},
            "smooth": {"type": "continuous", "roundness": 0.2},
            "length": """ + str(edge_length) + """
        },
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 200},
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -1500,
                "centralGravity": 0.005,
                "springLength": """ + str(edge_length) + """,
                "damping": 0.96,
                "avoidOverlap": 1
            }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true
        }
    }
    """
    net.set_options(options)
    
    # Use cached metrics
    degrees, out_degrees, node_labels, max_degree = compute_graph_metrics(G)
    
    # Use cached search matches
    matching_nodes = compute_search_matches(G, search_term, node_labels)
    
    import math
    
    for node in G.nodes():
        label = node_labels[node]
        if len(label) > 50:
            label = label[:50] + "..."
        degree = degrees[node]
        
        if max_degree > 0:
            normalized_size = math.log(1 + degree) / math.log(1 + max_degree)
            size = (15 + normalized_size * 45) * node_size_multiplier
        else:
            size = 20 * node_size_multiplier
        font_size = max(9, min(11, int(size / 4)))
        
        if out_degrees:
            title = f"Node: {label}\nIn-degree: {degree}\nOut-degree: {out_degrees[node]}"
        else:
            title = f"Node: {label}\nDegree: {degree}"
        
        node_matches_search = node in matching_nodes
        
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
        
        net.add_node(node, 
                    label=label,
                    title=title,
                    size=size,
                    color={'background': color, 'border': border_color},
                    borderWidth=border_width,
                    font={'size': font_size, 'bold': node_matches_search})
    
    # Pre-compute matching edges for search highlighting
    highlighted_edges = set()
    if search_term and matching_nodes:
        for edge in edges_to_render:
            if edge[0] in matching_nodes or edge[1] in matching_nodes:
                highlighted_edges.add(edge)
    
    # Add edges
    for edge in edges_to_render:
        if edge in highlighted_edges:
            net.add_edge(edge[0], edge[1], width=edge_width * 3, color={'color': '#FF6600', 'opacity': 0.9})
        else:
            net.add_edge(edge[0], edge[1], width=edge_width, color={'opacity': edge_opacity})
    
    return net, total_edges, len(edges_to_render)

# ============================================================================
# SUMMARIZER MODEL LOADING (lazy load to avoid conflicts)
# ============================================================================

@st.cache_resource
def load_summariser_model():
    try:
        from src.summariser import initialise_summariser, summarise_text
        summariser, tokeniser = initialise_summariser()
        return summariser, tokeniser
    except Exception as e:
        st.error(f"Could not load summarizer: {str(e)}")
        return None, None

# ============================================================================
# MAIN APP - SINGLE PAGE LAYOUT
# ============================================================================

# Initialize session state for graph loading
if 'graph_loaded' not in st.session_state:
    st.session_state.graph_loaded = False
if 'loaded_graph' not in st.session_state:
    st.session_state.loaded_graph = None

# ============================================================================
# SECTION 1: CITATION NETWORK GRAPH
# ============================================================================

st.header("Legal Map")

# Sidebar for graph options
st.sidebar.header("📊 Graph Options")

# Check if citation_graph.pkl exists
auto_loaded_graph = load_citation_graph()

if auto_loaded_graph is not None:
    st.sidebar.success("Legal Map has been found!")
    use_auto_graph = st.sidebar.checkbox("Use Default Legal Map", value=True)
else:
    use_auto_graph = False
    st.sidebar.info("Error: Legal Map has been found.")

uploaded_graph = st.sidebar.file_uploader(
    "Or upload a different NetworkX Pickle File", 
    type=['pickle', 'pkl', 'gpickle']
)

# Load Legal Map Button
if st.sidebar.button("🗺️ Load Legal Map", type="primary", use_container_width=True):
    if uploaded_graph is not None:
        st.session_state.loaded_graph = load_graph_from_pickle(uploaded_graph)
        st.session_state.graph_loaded = True
    elif use_auto_graph and auto_loaded_graph is not None:
        st.session_state.loaded_graph = auto_loaded_graph
        st.session_state.graph_loaded = True
    else:
        st.sidebar.error("Please select a graph source first!")

# Sidebar settings
if st.session_state.graph_loaded and st.session_state.loaded_graph is not None:
    st.sidebar.header("🎨 Interactive Features")
    highlight_colour_in = st.sidebar.color_picker("In-Degree Colour (Laws Inside)", "#00FF00")
    highlight_colour_out = st.sidebar.color_picker("Out-Degree Colour (Referenced By)", "#FF00FF")
    
    st.sidebar.header("⚙️ Visualisation Settings")
    node_size_multiplier = st.sidebar.slider("Node Size Multiplier", 1, 10, 3)
    edge_width = st.sidebar.slider("Edge Width", 1, 5, 1)
    arrow_size = st.sidebar.slider("Arrow Size", 5, 30, 8)
    edge_length = st.sidebar.slider("Edge Length", 500, 5000, 2000, step=100)
    
    st.sidebar.header("⚡ Performance Settings")
    max_edges_display = st.sidebar.number_input("Max Edges to Display (0 = all)", 
                                                min_value=0, max_value=50000, value=2000, step=500,
                                                help="For large graphs, try 1000-2000 for fastest loading")
    edge_opacity = st.sidebar.slider("Edge Opacity", 0.1, 1.0, 0.2, 0.1,
                                    help="Lower opacity = faster rendering")
    search_term = ""
else:
    highlight_colour_in = "#00FF00"
    highlight_colour_out = "#FF00FF"
    node_size_multiplier = 3
    edge_width = 1
    arrow_size = 8
    edge_length = 2000
    max_edges_display = 2000
    edge_opacity = 0.2
    search_term = ""

# Main graph display with search in right panel
graph_col1, graph_col2 = st.columns([2, 1])

with graph_col2:
    st.subheader("🔍 Search NSW Laws")
    
    if st.session_state.graph_loaded and st.session_state.loaded_graph is not None:
        temp_G = st.session_state.loaded_graph
        
        # Get all labels for search
        all_labels = []
        label_to_node = {}
        for node in temp_G.nodes():
            label = temp_G.nodes[node].get('label', str(node))
            all_labels.append(label)
            label_to_node[label] = node
        all_labels = sorted(set(all_labels))
        
        # Direct search with autocomplete
        search_input = st.text_input(
            "Search by name:",
            placeholder="Start typing law name...",
            key="node_search_input"
        )
        
        search_term = ""
        selected_node_for_info = None
        
        if search_input:
            # Find exact and partial matches
            exact_match = None
            partial_matches = []
            
            for label in all_labels:
                if search_input.lower() == label.lower():
                    exact_match = label
                    break
                elif search_input.lower() in label.lower():
                    partial_matches.append(label)
            
            if exact_match:
                search_term = exact_match
                selected_node_for_info = label_to_node[exact_match]
                st.success(f"✓ Found: **{exact_match}**")
            elif partial_matches:
                st.markdown("**Matching laws:**")
                selected = st.selectbox(
                    "Select a law:",
                    options=partial_matches[:10],
                    key="match_selector",
                    label_visibility="collapsed"
                )
                if selected:
                    search_term = selected
                    selected_node_for_info = label_to_node[selected]
            else:
                st.info("No matches found")
        
        # Display selected law information
        if selected_node_for_info is not None:
            st.markdown("---")
            G = temp_G
            node_id = selected_node_for_info
            node_label = G.nodes[node_id].get('label', str(node_id))
            
            # Display category if available
            category = G.nodes[node_id].get('category', 'N/A')
            st.markdown(f"**Category:** {category}")
            
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                in_deg = G.in_degree(node_id)
                out_deg = G.out_degree(node_id)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Laws Inside", in_deg)
                with col_b:
                    st.metric("Laws It Cites", out_deg)
                
                # Most cited law (highest in-degree from predecessors)
                st.markdown("---")
                st.markdown("**📊 Top Referenced Law**")
                predecessors = list(G.predecessors(node_id))
                if predecessors:
                    pred_degrees = [(pred, G.in_degree(pred)) for pred in predecessors]
                    pred_degrees.sort(key=lambda x: x[1], reverse=True)
                    top_pred_id, top_pred_deg = pred_degrees[0]
                    top_pred_label = G.nodes[top_pred_id].get('label', str(top_pred_id))
                    st.markdown(f"*Most cited law that cites this:*")
                    st.markdown(f"**{top_pred_label}**")
                    st.caption(f"Cited by {top_pred_deg} other laws")
                else:
                    st.info("No laws cite this one")
                
                # Highest law cited by it (highest out-degree from successors)
                st.markdown("---")
                st.markdown("**📤 Top Law It Cites**")
                successors = list(G.successors(node_id))
                if successors:
                    succ_degrees = [(succ, G.out_degree(succ)) for succ in successors]
                    succ_degrees.sort(key=lambda x: x[1], reverse=True)
                    top_succ_id, top_succ_deg = succ_degrees[0]
                    top_succ_label = G.nodes[top_succ_id].get('label', str(top_succ_id))
                    st.markdown(f"*Most influential law it references:*")
                    st.markdown(f"**{top_succ_label}**")
                    st.caption(f"Cites {top_succ_deg} other laws")
                else:
                    st.info("This law doesn't cite others")
            else:
                # Undirected graph
                deg = G.degree(node_id)
                st.metric("Connections", deg)
                
                neighbors = list(G.neighbors(node_id))
                if neighbors:
                    st.markdown("---")
                    st.markdown("**🔗 Top Connected Law**")
                    neighbor_degrees = [(n, G.degree(n)) for n in neighbors]
                    neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
                    top_n_id, top_n_deg = neighbor_degrees[0]
                    top_n_label = G.nodes[top_n_id].get('label', str(top_n_id))
                    st.markdown(f"**{top_n_label}**")
                    st.caption(f"{top_n_deg} connections")
        else:
            st.info("🔍 Search for a law to see details")
    else:
        st.info("Load a graph to search")

with graph_col1:
    if st.session_state.graph_loaded and st.session_state.loaded_graph is not None:
        G = st.session_state.loaded_graph
        
        graph_type = "Directed" if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)) else "Undirected"
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        st.info(f"📊 {graph_type} Graph: {num_nodes} nodes, {num_edges} edges")
        
        if num_edges > 10000:
            st.warning("💡 **Tip for faster loading**: Set 'Max Edges to Display' to 1000-2000")
        
        with st.spinner('Generating graph visualisation...'):
            net, total_edges, rendered_edges = create_pyvis_graph(
                G, search_term, node_size_multiplier, edge_width, arrow_size, 
                edge_length, max_edges_display, edge_opacity,
                highlight_colour_in, highlight_colour_out
            )
        
        if rendered_edges < total_edges:
            st.info(f"📉 Rendering {rendered_edges:,} of {total_edges:,} edges")
        else:
            st.success(f"✅ Rendering all {total_edges:,} edges")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
            net.save_graph(f.name)
            with open(f.name, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()
            
            # Disable physics after 5 seconds
            physics_disable_script = """
            <script type="text/javascript">
                setTimeout(function() {
                    if (typeof network !== 'undefined') {
                        network.setOptions({ physics: { enabled: false } });
                        console.log('Physics disabled after 5 seconds');
                    }
                }, 5000);
            </script>
            """
            html_content = html_content.replace('</body>', physics_disable_script + '</body>')
            
            # Click highlighting script
            highlighting_script = f"""
            <script type="text/javascript">
                window.addEventListener('load', function() {{
                    setTimeout(function() {{
                        if (typeof network === 'undefined') return;
                        var allNodes = network.body.data.nodes;
                        var allEdges = network.body.data.edges;
                        var originalNodeData = {{}};
                        var originalEdgeData = {{}};
                        
                        allNodes.forEach(function(node) {{
                            originalNodeData[node.id] = {{
                                color: {{background: node.color.background, border: node.color.border}},
                                borderWidth: node.borderWidth || 2
                            }};
                        }});
                        
                        allEdges.forEach(function(edge) {{
                            originalEdgeData[edge.id] = {{
                                color: edge.color || {{color: '#848484', opacity: {edge_opacity}}},
                                width: edge.width || 1
                            }};
                        }});
                        
                        function resetAll() {{
                            var nodeUpdates = [];
                            var edgeUpdates = [];
                            
                            allNodes.forEach(function(node) {{
                                if (originalNodeData[node.id]) {{
                                    nodeUpdates.push({{
                                        id: node.id,
                                        color: originalNodeData[node.id].color,
                                        borderWidth: originalNodeData[node.id].borderWidth
                                    }});
                                }}
                            }});
                            
                            allEdges.forEach(function(edge) {{
                                if (originalEdgeData[edge.id]) {{
                                    edgeUpdates.push({{
                                        id: edge.id,
                                        color: originalEdgeData[edge.id].color,
                                        width: originalEdgeData[edge.id].width
                                    }});
                                }}
                            }});
                            
                            if (nodeUpdates.length > 0) allNodes.update(nodeUpdates);
                            if (edgeUpdates.length > 0) allEdges.update(edgeUpdates);
                        }}
                        
                        network.on("selectNode", function(params) {{
                            resetAll();
                            setTimeout(function() {{
                                var nodeId = params.nodes[0];
                                var connectedEdges = network.getConnectedEdges(nodeId);
                                var inNodes = [];
                                var outNodes = [];
                                var edgeUpdates = [];
                                var nodeUpdates = [];
                                
                                connectedEdges.forEach(function(edgeId) {{
                                    var edge = allEdges.get(edgeId);
                                    if (!edge) return;
                                    var origWidth = originalEdgeData[edgeId] ? originalEdgeData[edgeId].width : 1;
                                    
                                    if (edge.to === nodeId) {{
                                        inNodes.push(edge.from);
                                        edgeUpdates.push({{
                                            id: edgeId,
                                            color: {{color: '{highlight_colour_in}', opacity: 0.95}},
                                            width: origWidth * 3
                                        }});
                                    }} else if (edge.from === nodeId) {{
                                        outNodes.push(edge.to);
                                        edgeUpdates.push({{
                                            id: edgeId,
                                            color: {{color: '{highlight_colour_out}', opacity: 0.95}},
                                            width: origWidth * 3
                                        }});
                                    }}
                                }});
                                
                                nodeUpdates.push({{
                                    id: nodeId,
                                    color: {{background: '#FFD700', border: '#FF8C00'}},
                                    borderWidth: 5
                                }});
                                
                                inNodes.forEach(function(nId) {{
                                    nodeUpdates.push({{
                                        id: nId,
                                        color: {{background: '{highlight_colour_in}', border: '#00CC00'}},
                                        borderWidth: 4
                                    }});
                                }});
                                
                                outNodes.forEach(function(nId) {{
                                    nodeUpdates.push({{
                                        id: nId,
                                        color: {{background: '{highlight_colour_out}', border: '#CC00CC'}},
                                        borderWidth: 4
                                    }});
                                }});
                                
                                if (edgeUpdates.length > 0) allEdges.update(edgeUpdates);
                                if (nodeUpdates.length > 0) allNodes.update(nodeUpdates);
                            }}, 10);
                        }});
                        
                        network.on("deselectNode", resetAll);
                        network.on("click", function(params) {{
                            if (params.nodes.length === 0 && params.edges.length === 0) resetAll();
                        }});
                    }}, 1000);
                }});
            </script>
            """
            html_content = html_content.replace('</body>', highlighting_script + '</body>')
            
            components.html(html_content, height=650)
        
        try:
            os.unlink(f.name)
        except:
            pass
    else:
        st.info("👈 Click **'Load Legal Map'** in the sidebar to visualize the citation network.")
        st.markdown("""
        ### How to use:
        1. The app will use **citation_graph.pkl** if it exists in the root directory
        2. Or upload your own NetworkX pickle file using the file uploader
        3. Click **'Load Legal Map'** to display the graph
        4. Use the search and visualization controls
        """)

st.markdown("---")

# ============================================================================
# SECTION 2: DOCUMENT SUMMARISER
# ============================================================================

st.header("Document Summariser")

# Load summarizer model
summariser, tokeniser = load_summariser_model()

# LAYOUT DEFINITION
main_col, controls_col = st.columns([4, 1])

# MAIN APPLICATION AREA & CONTROLS
with main_col:
    st.subheader("Enter your legal text")
    input_text = st.text_area("Paste the text from a legal document or legislation below:", height=200, label_visibility="collapsed")
    
    st.subheader("Or upload a PDF")
    uploaded_file = st.file_uploader("Upload a PDF document:", type=['pdf'], label_visibility="collapsed")

with controls_col:
    st.subheader("Controls")
    prompt_key = st.selectbox(
        "Summary style",
        options=list(PROMPT_OPTIONS.keys()),
        key="prompt_key"
    )
    
    st.markdown("---")
    
    api_provider = st.selectbox("Choose AI provider", ("Anthropic", "OpenAI", "Google"), key="api_provider")

    if api_provider == "Anthropic":
        model_name = st.selectbox("Choose a model", ("claude-3-opus-20240229", "claude-4-opus-20250925", "claude-3-5-sonnet-20240620", "claude-4-haiku-20250925"), key="model_name")
    elif api_provider == "OpenAI":
        model_name = st.selectbox("Choose a model", ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"), key="model_name")
    else: # Google
        model_name = st.selectbox("Choose a model", ("gemini-2.5-pro", "gemini-2.5-flash"), key="model_name")

    st.markdown("---")

    if st.button("Translate to plain English", type="primary", use_container_width=True):
        text_to_process = None
        if uploaded_file:
            text_to_process = extract_text_from_pdf(uploaded_file)
        elif input_text:
            text_to_process = input_text

        if isinstance(text_to_process, str) and text_to_process.strip():
            with st.spinner("Stage 1/2: Performing initial summary"):
                initial_summary = summarise_text(text_to_process, summariser, tokeniser)

            if initial_summary and "error" not in initial_summary.lower():
                
                # Combine original text and summary for Stage 2 API call
                stage2_input = f"""
ORIGINAL DOCUMENT:
---
{text_to_process}
---

SUMMARY OF DOCUMENT:
---
{initial_summary}
---
"""
                
                with st.spinner(f"Stage 2/2: Rewriting with {api_provider}..."):
                    final_translation = llm_handler.call_anthropic(PROMPT_OPTIONS[prompt_key], stage2_input, model_name) if api_provider == "Anthropic" else \
                                       llm_handler.call_openai(PROMPT_OPTIONS[prompt_key], stage2_input, model_name) if api_provider == "OpenAI" else \
                                       llm_handler.call_google(PROMPT_OPTIONS[prompt_key], stage2_input, model_name)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().strftime("%I:%M %p, %d %b %Y"),
                    'input_text': text_to_process[:200] + "..." if len(text_to_process) > 200 else text_to_process,
                    'initial_summary': initial_summary,
                    'final_output': final_translation if final_translation else None,
                    'prompt_style': prompt_key,
                    'provider': api_provider,
                    'model': model_name
                })
                
                st.rerun()
            else:
                st.error("Local summarisation failed")
        else:
            st.warning("Give me something to work with here!")
    
    st.markdown("---")
    
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ============================================================================
# section 3: Legalese Chat Bot Results 
# ============================================================================
st.markdown("---")
st.subheader("LegalEase Bot Chat")

if len(st.session_state.chat_history) == 0:
    st.info("No summaries yet. Enter text or upload a PDF above to get started!")
else:
    # Display chat history in reverse order (newest first)
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        # User message (input)
        col_user1, col_user2 = st.columns([1, 6])
        with col_user1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            try:
                st.image("static/person_a.png", width=50)
            except:
                st.markdown("**👤**")
            st.markdown("<p style='font-size: 12px; margin-top: -5px; font-weight: bold;'>User</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_user2:
            st.caption(f"🕐 {chat['timestamp']}")
            st.markdown(f"""<div style='background-color: #a8b0ba; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: inline-block; max-width: 100%;'>
                <strong>Input:</strong> {chat['input_text']}
            </div>""", unsafe_allow_html=True)
        
        # Assistant message (output)
        col_assist1, col_assist2 = st.columns([1, 6])
        with col_assist1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            try:
                st.image("static/ai_a.png", width=50)
            except:
                st.markdown("**⚖️**")
            st.markdown("<p style='font-size: 12px; margin-top: -5px; font-weight: bold;'>LegalEase Bot</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_assist2:
            st.caption(f"📊 {chat['provider']} ({chat['model']}) • Style: {chat['prompt_style']}")
            
            # Check if output is None or empty
            if chat['final_output'] is None or (isinstance(chat['final_output'], str) and not chat['final_output'].strip()):
                st.markdown(f"""<div style='background-color: #8fb8cc; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: inline-block; max-width: 100%;'>
                    <strong>Summary:</strong><br>
                    <span style='color: #d32f2f;'>Error: LegalEase failed to connect to {chat['provider']}.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style='background-color: #8fb8cc; padding: 15px; border-radius: 10px; margin-bottom: 10px; display: inline-block; max-width: 100%;'>
                    <strong>Summary:</strong><br>
                    {chat['final_output']}
                </div>""", unsafe_allow_html=True)
            
            # Expandable section for initial summary
            with st.expander("🔍 Show initial BART summary"):
                st.write(chat['initial_summary'])
        
        # Separator between conversations
        if idx < len(st.session_state.chat_history) - 1:
            st.markdown("---")
# ============================================================================
# DISCLAIMER & FOOTER
# ============================================================================

st.markdown("---")
st.info("DISCLAIMER: THIS TOOL PROVIDES SUMMARIES FOR INFORMATIONAL PURPOSES ONLY AND DOES NOT CONSTITUTE LEGAL ADVICE. IT IS NOT A SUBSTITUTE FOR A QUALIFIED LEGAL PROFESSIONAL.")
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip**: Node attributes `label` and `info` will be displayed automatically if present in your graph.")