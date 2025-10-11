# streamlit_app_UNITED.py
# Description - -------------
# There are two main parts to this page: The Legal Graph Function Visualisation and the Document Translator 
# The "UNITED" is more an indication of the integration of the two main developed features in development that were intially developed seperatly.
# Work was conducted to integrate the two code base into a more cohesive page that was user friendly, a challenge detailed in our report 
# -------------

import streamlit as st
from datetime import datetime
import math
import networkx as nx
import pickle
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import random
import re

# Import project modules
from src.pdf_handler import extract_text_from_pdf
from src.summariser import initialise_summariser, summarise_text
from src import llm_handler
from prompts import PROMPT_OPTIONS

# PAGE CONFIG (MUST BE FIRST)
st.set_page_config(page_title="LegalEase NSW", page_icon="⚖️", layout="wide")

# SESSION STATE INITIALISATION
defaults = {
    'history': [], 'loaded_graph': None, 'search_term': "", 'graph_initialised': False,
    'graph_html': None, 'show_full_graph': False, 'node_count': 80, 'show_node_slider': False,
    'pending_legislation_search': None, 'latest_translation': None, 'show_translator': True,
    'search_suggestions': [],
    'processing_translation': False,
    'prompt_key': 'With example' 
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# HELPER & CACHED FUNCTIONS
# Load custom CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# Load summariser model and cache it
# This summarisation model is the BART model CNN that was tested on our given legal documents.
@st.cache_resource
def load_summariser_model():
    try:
        return initialise_summariser()
    except Exception as e:
        st.error(f"Fatal Error: Could not load summariser model. {e}")
        return None, None

# Load citation graph from pickle file and cache it
@st.cache_data
def load_citation_graph():
    try:
        script_dir = Path(__file__).parent
        with open(script_dir / "citation_graph.pkl", "rb") as f:
            G = pickle.load(f)
            if isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return G
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"Could not auto-load citation_graph.pkl: {str(e)}")
    return None

# Compute graph metrics and cache them
@st.cache_data(show_spinner=False)
def compute_graph_metrics(_G):
    if isinstance(_G, (nx.DiGraph, nx.MultiDiGraph)):
        in_degrees = dict(_G.in_degree())
        out_degrees = dict(_G.out_degree())
    else:
        in_degrees = dict(_G.degree())
        out_degrees = None
    node_labels = {node: _G.nodes[node].get('label', str(node)) for node in _G.nodes()}
    return in_degrees, out_degrees, node_labels

# Extract year from legislation label
def extract_year_from_label(label):
    match = re.search(r'\b(19|20)\d{2}\b', label)
    return int(match.group()) if match else None

# Extract legislation name from text (e.g. 'RESIDENTIAL TENANCIES ACT 2010')
def extract_legislation_from_text(text):
    patterns = [
        r'([A-Z][A-Z\s&]+ACT\s+\d{4}(?:\s*\([A-Z]+\))?)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Act\s+\d{4}(?:\s*\([A-Z]+\))?)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()
    return None

# Find node in graph matching legislation name
def find_matching_node_in_graph(_G, legislation_name):
    if not legislation_name:
        return None
    
    node_labels = {node: _G.nodes[node].get('label', str(node)) for node in _G.nodes()}
    clean_name = re.sub(r'\s*\([A-Z]+\)\s*', '', legislation_name.upper().strip())
    
    for node, label in node_labels.items():
        clean_label = re.sub(r'\s*\([A-Z]+\)\s*', '', label.upper().strip())
        if clean_name == clean_label:
            return label
    
    name_words = set(clean_name.split()) - {'ACT', 'THE', 'OF', 'AND', 'FOR', 'IN', 'ON', 'AT'}
    best_match, best_score = None, 0
    
    for node, label in node_labels.items():
        clean_label = re.sub(r'\s*\([A-Z]+\)\s*', '', label.upper().strip())
        label_words = set(clean_label.split()) - {'ACT', 'THE', 'OF', 'AND', 'FOR', 'IN', 'ON', 'AT'}
        
        if name_words and label_words:
            score = len(name_words & label_words) / min(len(name_words), len(label_words))
            if score > best_score and score >= 0.7:
                best_score, best_match = score, label
    
    return best_match

# Convert NetworkX graph to PyVis with clear relationships
# BY Alistair Schillert, these functiosn are here to help.
def create_pyvis_graph(_G, search_term="", node_limit=80):
    net = Network(
        height='700px', width='100%', bgcolor='#fafafa', font_color='#333333',
        directed=isinstance(_G, (nx.DiGraph, nx.MultiDiGraph)),
        filter_menu=False, select_menu=False, notebook=False
    )

    net.set_options("""
    {
        "nodes": {
            "font": {"size": 12, "face": "Arial", "color": "#000000", "bold": {"color": "#000000"}},
            "shape": "dot", "borderWidth": 2, "borderWidthSelected": 3, "shadow": false
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 1.0, "type": "arrow"}},
            "color": {"color": "#d0d0d0", "highlight": "#FF6600", "hover": "#FF8800"},
            "smooth": {"enabled": true, "type": "continuous", "roundness": 0.5},
            "width": 2, "selectionWidth": 4
        },
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
                "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 200,
                "springConstant": 0.08, "damping": 0.95, "avoidOverlap": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": true, "iterations": 200, "updateInterval": 20, "fit": true},
            "minVelocity": 0.75, "maxVelocity": 15
        },
        "interaction": {
            "hover": true, "navigationButtons": true, "keyboard": true, "zoomView": true,
            "dragView": true, "tooltipDelay": 100, "hideEdgesOnDrag": false,
            "hideEdgesOnZoom": false, "zoomSpeed": 1
        }
    }
    """)
    
    in_degrees, out_degrees, node_labels = compute_graph_metrics(_G)
    matching_nodes = {node for node, label in node_labels.items() 
                     if search_term and search_term.lower() in label.lower()}
    
    nodes_to_show = set()
    depth = 2 if search_term and matching_nodes else 0
    
    if search_term and matching_nodes:
        for match_node in matching_nodes:
            nodes_to_show.add(match_node)
            
            if isinstance(_G, (nx.DiGraph, nx.MultiDiGraph)):
                predecessors = set(_G.predecessors(match_node))
                nodes_to_show.update(predecessors)
                successors = set(_G.successors(match_node))
                nodes_to_show.update(successors)
                
                if depth > 1 and len(nodes_to_show) < 150:
                    for pred in list(predecessors)[:20]:
                        nodes_to_show.update(list(_G.predecessors(pred))[:5])
                    for succ in list(successors)[:20]:
                        nodes_to_show.update(list(_G.successors(succ))[:5])
            else:
                nodes_to_show.update(_G.neighbors(match_node))
        
        if len(nodes_to_show) > 150:
            neighbours = nodes_to_show - matching_nodes
            top_neighbours = sorted(neighbours, key=lambda n: in_degrees.get(n, 0), reverse=True)[:100]
            nodes_to_show = matching_nodes | set(top_neighbours)
    else:
        # Use the node_limit parameter from slider
        top_nodes = sorted(_G.nodes(), key=lambda n: in_degrees.get(n, 0), reverse=True)[:node_limit]
        nodes_to_show = set(top_nodes)
    
    max_degree = max(in_degrees.values()) if in_degrees else 1
    
    # Pre-calculate top citing and referenced laws for each node
    top_citing_laws, top_referenced_laws = {}, {}
    
    for node in nodes_to_show:
        # Find top 3 laws that cite this node the most
        if isinstance(_G, (nx.DiGraph, nx.MultiDiGraph)):
            predecessors = list(_G.predecessors(node))
            if predecessors:
                # Sort predecessors by their out-degree (most active citers)
                sorted_citers = sorted(predecessors, key=lambda n: out_degrees.get(n, 0), reverse=True)[:3]
                top_citing_laws[node] = [node_labels.get(n, str(n)) for n in sorted_citers]
            
            # Find top 3 laws that this node references the most
            successors = list(_G.successors(node))
            if successors:
                # Sort successors by their in-degree (most referenced laws)
                sorted_refs = sorted(successors, key=lambda n: in_degrees.get(n, 0), reverse=True)[:3]
                top_referenced_laws[node] = [node_labels.get(n, str(n)) for n in sorted_refs]
    
    for node in nodes_to_show:
        label = node_labels[node]
        degree = in_degrees.get(node, 0)
        display_label = label if len(label) <= 40 else label[:37] + "..."
        
        # Build tooltip with law name and lists
        tooltip_lines = [f"Law Name: {label}", ""]
        
        # Laws that cite this the most
        tooltip_lines.append("Laws that Cite This The Most:")
        if node in top_citing_laws:
            for i, citing_law in enumerate(top_citing_laws[node], 1):
                citing_law_short = citing_law if len(citing_law) <= 50 else citing_law[:47] + "..."
                tooltip_lines.append(f"{i}. {citing_law_short}")
        else:
            tooltip_lines.append("None")
        tooltip_lines.append("")
        
        # Highest referenced laws
        tooltip_lines.append("Highest Referenced Laws:")
        if node in top_referenced_laws:
            for i, ref_law in enumerate(top_referenced_laws[node], 1):
                ref_law_short = ref_law if len(ref_law) <= 50 else ref_law[:47] + "..."
                tooltip_lines.append(f"{i}. {ref_law_short}")
        else:
            tooltip_lines.append("None")
        
        title = "\n".join(tooltip_lines)
        is_match = node in matching_nodes
        node_size = 12 + min(math.log(1 + degree) * 8, 35)
        
        if is_match:
            node_colour, border_colour, border_width = '#FFD700', '#FF8C00', 4
        else:
            intensity = min(degree / (max_degree * 0.3), 1.0)
            r = int(135 + (30 - 135) * intensity)
            g = int(206 + (144 - 206) * intensity)
            b = int(250 + (255 - 250) * intensity)
            node_colour, border_colour, border_width = f'rgb({r},{g},{b})', '#2B7CE9', 2
        
        net.add_node(
            node, label=display_label, title=title, size=node_size,
            color={
                'background': node_colour, 'border': border_colour,
                'highlight': {'background': '#FF6600', 'border': '#CC5200'},
                'hover': {'background': '#FFD700', 'border': '#FF8C00'}
            },
            borderWidth=border_width,
            font={'size': 14 if is_match else 11, 'color': '#000000'}
        )
    
    edges_to_add = []
    for u, v in _G.edges():
        if u in nodes_to_show and v in nodes_to_show:
            is_highlighted = u in matching_nodes or v in matching_nodes
            priority = (100 if is_highlighted else 0) + in_degrees.get(u, 0) + in_degrees.get(v, 0)
            edges_to_add.append((u, v, is_highlighted, priority))
    
    edges_to_add.sort(key=lambda x: x[3], reverse=True)
    # Scale max edges based on node count
    max_edges = int(node_limit * 3)
    
    for u, v, is_highlighted, _ in edges_to_add[:max_edges]:
        edge_colour = '#FF6600' if is_highlighted else '#d0d0d0'
        edge_width = 3 if is_highlighted else 1.5
        net.add_edge(u, v, width=edge_width, color=edge_colour,
                    title=f"{node_labels[u]}<br>➡️ cites ➡️<br>{node_labels[v]}")
    
    html_string = net.generate_html()
    
    # Inject JavaScript to set initial zoom level and calm physics after stabilisation
    zoom_script = """
    <script type="text/javascript">
        network.once("stabilizationIterationsDone", function() {
            network.moveTo({scale: 1.1, animation: {duration: 800, easingFunction: 'easeInOutQuad'}});
            network.setOptions({ 
                physics: { 
                    forceAtlas2Based: {
                        gravitationalConstant: -20, centralGravity: 0.005,
                        springLength: 200, springConstant: 0.02, damping: 0.98
                    }
                } 
            });
        });
    </script>
    """
    html_string = html_string.replace('</body>', zoom_script + '</body>')
    return html_string, len(nodes_to_show), len(edges_to_add[:max_edges])

# RENDER UI - GRID LAYOUT
load_css("static/style.css")

# Introduced search bug eep
# summariser, tokeniser = load_summariser_model()

# Load citation graph
if st.session_state.loaded_graph is None:
    with st.spinner("Loading citation graph..."):
        st.session_state.loaded_graph = load_citation_graph()

# Custom CSS for tighter spacing and alignment
st.markdown("""
    <style>
    .header-title {
        font-size: 25px; font-weight: bold; color: #2c3e50;
        letter-spacing: 0.5px; line-height: 1.3; margin-bottom: 3px;
    }
    .header-subtitle {font-size: 18px; color: #7f8c8d; font-weight: normal;}
    .stTextInput > label {margin-bottom: 0px !important; padding-bottom: 2px !important;}
    .history-scroll {max-height: 400px; overflow-y: auto;}
    div[data-testid="stVerticalBlock"] > div:has(h3:contains("Translation History")) {margin-top: -20px !important;}
    div[data-testid="column"]:has(h3:contains("Document simplifier")) {margin-bottom: -30px !important;}
    </style>
""", unsafe_allow_html=True)

# SITE HEADER + NSW LEGISLATION MAP HEADER
col1, col2 = st.columns([1, 1])

with col1:
    # Logo and title section
    logo_col, title_col = st.columns([1, 3])
    with logo_col:
        st.image("static/logo3.jpg", width=300)
    with title_col:
        st.markdown("""
            <div style='padding-top: 10px;'>
                <div class='header-title'>AI-powered legal document simplifier & citation network</div>
                <div class='header-subtitle'>Transforming NSW legislation into plain English</div>
            </div>
        """, unsafe_allow_html=True)

with col2:
    # NSW Legislation Map header
    st.subheader("NSW legislation map")
    
    if st.session_state.loaded_graph is not None:
        G = st.session_state.loaded_graph
        
        # Check if there's a pending legislation search from document processing
        if st.session_state.pending_legislation_search:
            matched_node = find_matching_node_in_graph(G, st.session_state.pending_legislation_search)
            if matched_node:
                st.session_state.search_term = matched_node
                st.session_state.graph_html = None
            else:
                st.warning(f"Could not find '{st.session_state.pending_legislation_search}' in the citation network.")
            st.session_state.pending_legislation_search = None
        
        if not st.session_state.graph_initialised:
            try:
                node_labels = {node: G.nodes[node].get('label', str(node)) for node in G.nodes()}
                modern_nodes = [node for node, label in node_labels.items() 
                               if (year := extract_year_from_label(label)) and year >= 2000]
                
                in_degrees = dict(G.in_degree() if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)) else G.degree())
                
                if modern_nodes:
                    top_modern = sorted(modern_nodes, key=lambda n: in_degrees.get(n, 0), reverse=True)[:30]
                    random_node = random.choice(top_modern)
                else:
                    top_nodes = sorted(G.nodes(), key=lambda n: in_degrees.get(n, 0), reverse=True)[:50]
                    random_node = random.choice(top_nodes)
                
                st.session_state.search_term = node_labels[random_node]
                st.session_state.graph_initialised = True
            except Exception as e:
                st.error(f"Error selecting random node: {str(e)}")
                st.session_state.search_term = ""
                st.session_state.graph_initialised = True
        
        # Aligned search controls in single row
        search_col1, search_col2 = st.columns([5.25, 0.75])
        
        with search_col1:
            st.markdown("""
                <style>
                div[data-testid="stTextInput"] > div > div > input {
                    background-color: #ffffff !important; font-weight: 400 !important;
                    border: 1px solid #dfe1e5 !important; border-radius: 24px !important;
                    padding: 10px 20px !important; font-size: 14px !important;
                    box-shadow: 0 1px 6px rgba(32,33,36,.28) !important;
                }
                div[data-testid="stTextInput"] > div > div > input:hover {
                    box-shadow: 0 1px 8px rgba(32,33,36,.35) !important;
                }
                div[data-testid="stTextInput"] > div > div > input:focus {
                    border: 1px solid #dfe1e5 !important;
                    box-shadow: 0 1px 8px rgba(32,33,36,.35) !important;
                    outline: none !important;
                }
                .suggestion-box {
                    background: white; border: 1px solid #dfe1e5; border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(32,33,36,.28); margin-top: -10px; padding: 0;
                }
                .suggestion-header {
                    padding: 12px 16px 8px 16px; font-size: 12px; font-weight: 500;
                    color: #70757a; text-transform: uppercase; letter-spacing: 0.5px;
                    border-bottom: 1px solid #f1f3f4;
                }
                .suggestion-item {
                    padding: 12px 16px; cursor: pointer; display: flex; align-items: center;
                    font-size: 14px; color: #202124; transition: background-color 0.1s;
                    border: none; border-bottom: 1px solid #f1f3f4;
                }
                .suggestion-item:last-child {border-bottom: none; border-radius: 0 0 8px 8px;}
                .suggestion-item:first-of-type {border-radius: 0;}
                .suggestion-item:hover {background-color: #f8f9fa;}
                .suggestion-icon {margin-right: 12px; color: #5f6368; font-size: 16px;}
                div[data-testid="column"] > div > div > div > button[kind="secondary"] {
                    border: none !important; border-radius: 0 !important; box-shadow: none !important;
                    padding: 12px 16px !important; margin: 0 !important; background: white !important;
                    color: #202124 !important; text-align: left !important; font-weight: 400 !important;
                }
                div[data-testid="column"] > div > div > div > button[kind="secondary"]:hover {
                    background: #f8f9fa !important; border: none !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            search_input = st.text_input(
                "Search legislation by name:", value=st.session_state.search_term,
                key='search_input', placeholder="Search NSW legislation...",
                label_visibility="collapsed"
            )
            
            # Generate search suggestions when user types
            # This idea is for the users trying to see exact laws with search 
            if search_input and search_input != st.session_state.search_term:
                # Find matching laws
                node_labels = {node: G.nodes[node].get('label', str(node)) for node in G.nodes()}
                matches = []
                search_lower = search_input.lower()
                
                for node, label in node_labels.items():
                    if search_lower in label.lower():
                        # Calculate relevance score
                        if label.lower().startswith(search_lower):
                            score = 100
                        elif label.lower().split()[0].startswith(search_lower):
                            score = 90
                        else:
                            score = 100 - label.lower().index(search_lower)
                        matches.append((label, score, node))
                
                matches.sort(key=lambda x: x[1], reverse=True)
                st.session_state.search_suggestions = [m[0] for m in matches[:3]]
            elif not search_input:
                st.session_state.search_suggestions = []
            
            # Display the suggestions 
            # This will give all the suggestions to each.
            if st.session_state.search_suggestions:
                st.markdown('<div class="suggestion-box"><div class="suggestion-header">Suggested Laws that match</div>', unsafe_allow_html=True)
                for idx, suggestion in enumerate(st.session_state.search_suggestions):
                    if st.button(f"🔍 {suggestion}", key=f"suggestion_{idx}", 
                                use_container_width=True, type="secondary"):
                        st.session_state.search_term = suggestion
                        st.session_state.graph_html = None
                        st.session_state.search_suggestions = []
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        with search_col2:
            st.write("")
            st.write("")
            
            # Toggle button for showing slider
            show_slider = st.checkbox("More laws?", value=st.session_state.show_node_slider,
                                     key="show_slider_toggle")
            
            # Update state if changed
            if show_slider != st.session_state.show_node_slider:
                st.session_state.show_node_slider = show_slider
                if not show_slider:
                    st.session_state.node_count = 80
                    st.session_state.graph_html = None
            
            # Show slider only if checkbox is checked
            if st.session_state.show_node_slider:
                total_nodes = G.number_of_nodes()
                node_count = st.slider(
                    "Laws to display:", min_value=50, max_value=min(total_nodes, 1000),
                    value=st.session_state.node_count, step=10, key="node_count_slider",
                    help=f"Adjust number of nodes (max: {min(total_nodes, 500)})"
                )
                
                if node_count != st.session_state.node_count:
                    st.session_state.node_count = node_count
                    st.session_state.graph_html = None
        
        if search_input != st.session_state.search_term:
            st.session_state.search_term = search_input
            st.session_state.graph_html = None


# ROW 2: Document simplifier (left) & NSW Legislation Map Graph (right)
# The idea for each is to have a more balanced page.
# Document Simplifier by Brett McKeehan
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Document simplifier")
    
    # Show translator input only if show_translator is True
    if st.session_state.show_translator:
        # Create two columns: 2/3 for text input, 1/3 for controls
        input_col, controls_col = st.columns([3, 1])
        
        with input_col:
            input_text = st.text_area(
                "Paste legal text:", height=500,
                placeholder="Add legal text that you'd like to read in plain English, select your preferred summary style and click 'Translate'",
                key="input_text_area"
            )
        
        with controls_col:
            uploaded_file = st.file_uploader("Upload PDF:", type=['pdf'], 
                                            label_visibility="visible", key="pdf_uploader")
            st.markdown("---")

            # Map display labels to actual prompt labels
            PROMPT_DISPLAY_LABELS = {
                "Helper mode": "With example",
                "More detail": "Default",
                "Explain like I'm 5": "Explain like I'm 5"
            }
    
            # Display labels in radio buttons
            selected_display_label = st.radio(
                "Simplification style:",
                options=list(PROMPT_DISPLAY_LABELS.keys()),
                index=0,  # Which option is default
                key="prompt_display"
            )

            # Store prompt key in session state
            st.session_state.prompt_key = PROMPT_DISPLAY_LABELS[selected_display_label]

            st.markdown("---")
            
            with st.form(key='translate_form', clear_on_submit=False):
                process_button = st.form_submit_button("Translate to plain English", 
                                                       use_container_width=True, type="primary")
            
        if process_button:
            # SET PROCESSING FLAG IMMEDIATELY
            st.session_state.processing_translation = True
            
            text_to_process = None
            if uploaded_file:
                with st.spinner("Extracting text from PDF..."):
                    text_to_process = extract_text_from_pdf(uploaded_file)
            elif input_text.strip():
                text_to_process = input_text

            if text_to_process:
                summariser, tokeniser = load_summariser_model() # Attempted fix for search bug     
                if summariser is None or tokeniser is None:
                    st.error("Cannot proceed without summariser model. Please check your model files.")
                    st.session_state.processing_translation = False
                else:
                    with st.spinner("Stage 1/2: Analysing document..."):
                        initial_summary = summarise_text(text_to_process, summariser, tokeniser)
                    
                    if initial_summary and "error" not in initial_summary.lower():
                        stage2_input = f"ORIGINAL DOCUMENT:\n---\n{text_to_process}\n---\n\nSUMMARY OF DOCUMENT:\n---\n{initial_summary}\n---"
                        
                        # Set default model and provider
                        selected_model = "gpt-4o"
                        selected_provider = "OpenAI"
                        selected_prompt = st.session_state.prompt_key
                        
                        with st.spinner(f"Stage 2/2: Rewriting with {selected_model}..."):
                            try:
                                final_translation = llm_handler.call_openai(
                                    PROMPT_OPTIONS[selected_prompt], stage2_input, selected_model
                                )
                                
                                if final_translation and len(final_translation.strip()) > 0:
                                    legislation_name = extract_legislation_from_text(text_to_process)
                                    if legislation_name:
                                        st.session_state.pending_legislation_search = legislation_name
                                    
                                    st.session_state.latest_translation = {
                                        'timestamp': datetime.now().strftime("%I:%M %p, %d %b %Y"),
                                        'input_text': text_to_process,
                                        'initial_summary': initial_summary,
                                        'final_output': final_translation,
                                        'prompt_style': selected_prompt,
                                        'model': f"{selected_provider} - {selected_model}",
                                        'legislation': legislation_name
                                    }
                                    
                                    st.session_state.history.insert(0, st.session_state.latest_translation.copy())
                                    st.session_state.show_translator = False
                                    
                                    # CLEAR PROCESSING FLAG BEFORE RERUN
                                    st.session_state.processing_translation = False
                                    st.rerun()
                                else:
                                    st.error("The AI model came up empty. Try again with different text or check your API key")
                                    st.info(f"Make sure your {selected_provider} API key is set up properly")
                                    st.session_state.processing_translation = False
                            except Exception as e:
                                st.error(f"AI model error: {str(e)}")
                                st.info("""Possible issues:
                                - Check your API key is valid
                                - Make sure you have API credits
                                - Try with shorter text
                                - Check your internet connection""")
                                st.session_state.processing_translation = False
                    else:
                        st.error("Stage 1 summarisation failed. Check your text format or try a different document.")
                        st.session_state.processing_translation = False
            else:
                st.warning("You need to add text or a PDF for us to translate")
                st.session_state.processing_translation = False
    
        # MORE PROMINENT DISCLAIMER
        st.info("**DISCLAIMER:** WE PROVIDE TRANSLATIONS FOR INFORMATIONAL PURPOSES ONLY AND ARE NO SUBSTITUTE FOR QUALIFIED LEGAL ADVICE")

    # Display latest translation when translator is hidden
    if not st.session_state.show_translator and st.session_state.latest_translation:
        if st.button("← NEW TRANSLATION", key="new_translation_btn", type="primary"):
            st.session_state.show_translator = True
            st.rerun()

        # MORE PROMINENT DISCLAIMER
        st.info("**DISCLAIMER:** WE PROVIDE TRANSLATIONS FOR INFORMATIONAL PURPOSES ONLY AND ARE NO SUBSTITUTE FOR QUALIFIED LEGAL ADVICE")
        item = st.session_state.latest_translation
        
        # User message (input)
        col_user1, col_user2 = st.columns([1, 8])
        with col_user1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            try:
                st.image("static/person_a.png", width=50)
            except:
                st.markdown("**👤**")
            st.markdown("<p style='font-size: 18px; margin-top: -5px; font-weight: bold;'>User Input</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_user2:
            st.caption(f"🕐 {item['timestamp']}")
            st.markdown(f"""<div style='background-color: #f5faf6; font-size: 18px; padding: 10px; border-radius: 5px; margin-bottom: 5px; display: inline-block; max-width: 100%;'>
                        <strong>Initial summary:</strong> {item['legislation']}
                    </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
  
        
        # Assistant message (output)
        col_assist1, col_assist2 = st.columns([1, 8])
        with col_assist1:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            try:
                st.image("static/ai_a.png", width=50)
            except:
                st.markdown("**⚖️**")
            st.markdown("<p style='font-size: 18px; margin-top: -5px; font-weight: bold;'>Translation Helper</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_assist2:
            st.caption(f"📊 {item['model']} • Style: {item['prompt_style']}")
            
            if item['final_output'] is None or (isinstance(item['final_output'], str) and not item['final_output'].strip()):
                st.markdown(f"""<div style='background-color: #fca19d; font-size: 18px; padding: 10px; border-radius: 5px; margin-bottom: 5px; display: inline-block; max-width: 100%;'>
                    <span style='color: #d32f2f;'>ERROR: LEGALEASE NSW FAILED TO GENERATE A TRANSLATION</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style='background-color: #edfcff; font-size: 18px; padding: 10px; border-radius: 5px; margin-bottom: 5px; display: inline-block; max-width: 100%;'>
                    {item['final_output'].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')}
                </div>""", unsafe_allow_html=True)
            
            # Expandable section for initial summary and original text
            with st.expander("🔍 Show original text"):
                st.markdown("**Original text**")
                st.text(item['input_text'])
                st.markdown("---")
                st.markdown("**Initial summary**")
                st.write(item['initial_summary'])

with col2:
    # Continue NSW Legislation map (display graph)
    if st.session_state.loaded_graph is not None:
        if st.session_state.graph_html is None:
            with st.spinner("Generating visualisation..."):
                try:
                    html_content, num_nodes, num_edges = create_pyvis_graph(
                        G, st.session_state.search_term, st.session_state.node_count
                    )
                    st.session_state.graph_html = html_content
                    st.session_state.graph_stats = {'nodes': num_nodes, 'edges': num_edges}
                except Exception as e:
                    st.error(f"Error generating graph: {str(e)}")
                    st.session_state.graph_html = "<html><body><h3>Error loading graph</h3></body></html>"
        
        if hasattr(st.session_state, 'graph_stats'):
            stats = st.session_state.graph_stats
            st.caption(f"Displaying {stats['nodes']} nodes and {stats['edges']} citation relationships")
        
        if st.session_state.graph_html:
            components.html(st.session_state.graph_html, height=700, scrolling=False)
            
            with st.expander("How to use this map"):
                total_nodes = G.number_of_nodes()
                total_edges = G.number_of_edges()
                st.markdown(f"""
                **Features of the Legal Map**

                - **Circle** → Represents an individual piece of NSW law  
                - **Arrow direction** → Shows citation flow (e.g. **A → B** means *Act A cites Act B*)

                ---

                **Interacting with the Visualisation**

                - **Drag** to pan around the network  
                - **Scroll** to zoom in/out  
                - **Click a node** to highlight its connections  
                - **Hover over nodes** to view full details  
                - **Hover over arrows** to see specific citation relationships  

                ---

                **Visual Guide**

                - **Node size** → Indicates influence (larger = cited more often)  
                - **Node colour** → Darker blue = more frequently cited  
                - **Gold nodes** → Match your search term  
                - **Orange arrows** → Citations involving your searched legislation  
                - **Grey arrows** → Other citation relationships  

                ---

                **Full Network Overview**

                This network contains **{total_nodes}** Acts with **{total_edges}** citation links between them.
                """)
                 
                # ORIGINAL DISCLAIMER POSITION
                # st.info("DISCLAIMER: THIS TOOL PROVIDES SUMMARIES FOR INFORMATIONAL PURPOSES ONLY AND IS NOT A SUBSTITUTE FOR QUALIFIED LEGAL ADVICE")
        else:
            st.warning("Graph visualisation failed to load.")
    else:
        st.warning("Could not find 'citation_graph.pkl'. The Legal Map feature will be unavailable.")

# TRANSLATION HISTORY DROPDOWN (full width, only show if history exists)
if len(st.session_state.history) > 0:
    st.markdown("<div style='margin-top: -20px;'></div>", unsafe_allow_html=True)
    st.subheader("Translation History")
    
    # Dropdown for chat history
    with st.expander(f"View all translations ({len(st.session_state.history)} items)", expanded=False):
        # Buttons row
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("Clear history", key="clear_history_btn", use_container_width=True):
                st.session_state.history = []
                st.session_state.latest_translation = None
                st.session_state.show_translator = True  # Add this line
                st.rerun()
        
        with btn_col2:
            # Create downloadable text file from history
            history_text = "LegalEase NSW - Translation History\n" + "=" * 50 + "\n\n"
            
            for idx, chat in enumerate(st.session_state.history):
                history_text += f"Translation #{len(st.session_state.history) - idx}\n"
                history_text += f"Timestamp: {chat['timestamp']}\n"
                if chat.get('legislation'):
                    history_text += f"Legislation: {chat['legislation']}\n"
                history_text += f"Model: {chat['model']}\n"
                history_text += f"Style: {chat['prompt_style']}\n"
                history_text += "-" * 50 + "\n\n"
                history_text += f"ORIGINAL INPUT:\n{chat['input_text']}\n\n"
                history_text += f"BART SUMMARY:\n{chat['initial_summary']}\n\n"
                history_text += f"PLAIN ENGLISH TRANSLATION:\n"
                history_text += (chat['final_output'] if chat['final_output'] else "Error: Translation failed.") + "\n\n"
                history_text += "=" * 50 + "\n\n"
            
            st.download_button(
                label="📥 Download", data=history_text,
                file_name=f"legalease_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain", key="download_history_btn", use_container_width=True
            )
        
        st.markdown("---")
        
        # Display chat history
        for idx, chat in enumerate(st.session_state.history):
            # User message (input)
            col_user1, col_user2 = st.columns([1, 20])
            with col_user1:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                try:
                    st.image("static/person_a.png", width=50)
                except:
                    st.markdown("**👤**")
                st.markdown("<p style='font-size: 18px; margin-top: -5px; font-weight: bold;'>User Input</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col_user2:
                st.caption(f"🕐 {chat['timestamp']}")
                if chat.get('legislation'):
                    st.caption(f"📋 {chat['legislation']}")
                st.markdown(f"""<div style='background-color: #f5faf6; padding: 5px; border-radius: 5px; margin-bottom: 5px; max-width: 100%; font-family: inherit; font-size: 18px; font-weight: normal;'>
                    {chat['input_text'].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')}
                </div>""", unsafe_allow_html=True)
            
            # Assistant message (output)
            col_assist1, col_assist2 = st.columns([1, 20])
            with col_assist1:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                try:
                    st.image("static/ai_a.png", width=50)
                except:
                    st.markdown("**⚖️**")
                st.markdown("<p style='font-size: 18px; margin-top: -5px; font-weight: bold;'>Translation Helper</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col_assist2:
                st.caption(f"📊 {chat['model']} • Style: {chat['prompt_style']}")
                
                if chat['final_output'] is None or (isinstance(chat['final_output'], str) and not chat['final_output'].strip()):
                    st.markdown(f"""<div style='background-color: #fca19d; font-size: 18px; padding: 10px; border-radius: 5px; margin-bottom: 5px; max-width: 100%;'>
                        <span style='color: #d32f2f;'>ERROR: LEGALEASE NSW FAILED TO GENERATE A TRANSLATION</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style='background-color: #edfcff; font-size: 18px; padding: 10px; border-radius: 5px; margin-bottom: 5px; max-width: 100%; font-weight: normal;'>
                        {chat['final_output'].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')}
                    </div>""", unsafe_allow_html=True)
                
                # Expandable section for initial summary
                with st.expander("🔍 Show initial summary"):
                    st.markdown("**Summary**")
                    st.write(chat['initial_summary'])
            
            # Separator between conversations
            if idx < len(st.session_state.history) - 1:
                st.markdown("---")