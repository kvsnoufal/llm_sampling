import os
import json
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Helper Functions

def format_node(node, is_first_child, is_root):
    """Format node with HTML based on its properties."""
    color = 'blue' if is_first_child or is_root else 'grey'
    font_style = 'font-size:9px;font-style:strong;vertical-align: sub;' if not is_root else ''
    return f"<span style='color:{color};'>{node['name']}</span>" \
           f"<span style='color:{color};{font_style}'> {int(node['probabilityScore']*100)}%</span>"

def traverse_tree(node, level, levels, is_first_child=False, is_root=False):
    """Recursively traverse and format the tree nodes by level."""
    if level not in levels:
        levels[level] = []
    levels[level].append(format_node(node, is_first_child, is_root))

    if 'children' in node and node['children']:
        for idx, child in enumerate(node['children']):
            traverse_tree(child, level + 1, levels, is_first_child=(idx == 0))

def get_tree_levels(node, is_greedy=True):
    """Get tree levels formatted for either greedy or sampling."""
    levels = {}
    traverse_tree(node, 0, levels, is_root=True)
    return levels

def save_tree(tree, filename):
    """Save tree structure to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(tree, f, indent=2)

def generate_text(model, tokenizer, input_text, num_tokens, temperature, top_p):
    """Generate text using the model and tokenizer."""
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    greedy_tree, sampling_tree = {"name": "<bos>", "children": []}, {"name": "<bos>", "children": []}
    current_node_greedy, current_node_sampling = greedy_tree, sampling_tree
    greedy_input_ids, sampling_input_ids = input_ids.input_ids.clone(), input_ids.input_ids.clone()
    
    next_line_counter_greedy, next_line_counter_sampling = 2, 2
    should_break_greedy, should_break_sampling = False, False
    
    OUTDIR = "output_v2"
    os.makedirs(OUTDIR, exist_ok=True)
    
    def update_tree_and_text():
        """Update and return the formatted tree text for both greedy and sampling methods."""
        greedy_levels = get_tree_levels(greedy_tree, is_greedy=True)
        sampling_levels = get_tree_levels(sampling_tree, is_greedy=False)

        def format_levels(levels):
            level_html = ""
            for level in range(len(levels)):
                if level in levels:
                    level_text = "".join(
                        f"<div style='padding:5px 0; border-bottom:1px solid grey; font-size:12px;'>{node}</div>"
                        for node in levels[level]
                    )
                    level_html += f"<div style='display:flex; flex-direction:column; align-items:flex-start; margin-right:2px; max-width:200px;'>{level_text}</div>"
            return f"<div style='display:flex; flex-direction:row; overflow-x:auto;'>{level_html}</div>"

        return format_levels(greedy_levels), format_levels(sampling_levels)

    with torch.no_grad():
        for _ in range(num_tokens):
            # Greedy Generation
            if not should_break_greedy:
                outputs = model(greedy_input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(logits, 10, dim=-1)
                top_k_probs = F.softmax(top_k_logits, dim=-1)[0].cpu().tolist()
                top_k_indices = top_k_indices[0].cpu().tolist()
                
                children = []
                for top_k_token_index, token_id in enumerate(top_k_indices):
                    if top_k_token_index == 0 and (token_id == 1 or next_line_counter_greedy > 2 or token_id == 13173):
                        should_break_greedy = True
                    
                    generated_token = tokenizer.convert_ids_to_tokens([token_id])[0].replace('▁', " ")
                    new_node = {"name": generated_token, "children": [], "probabilityScore": top_k_probs[top_k_token_index]}
                    children.append(new_node)
                
                current_node_greedy["children"].extend(children)
                if children:
                    current_node_greedy = children[0]
                
                if not should_break_greedy:
                    first_token_tensor = torch.tensor([top_k_indices[0]], device=greedy_input_ids.device).unsqueeze(-1)
                    greedy_input_ids = torch.cat([greedy_input_ids, first_token_tensor], dim=-1)
            
            # Sampling Generation
            if not should_break_sampling:
                outputs = model(sampling_input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')
                
                next_token_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                top_k_logits, top_k_indices = torch.topk(logits, 10, dim=-1)
                top_k_probs = F.softmax(top_k_logits, dim=-1)[0].cpu().tolist()
                top_k_indices = top_k_indices[0].cpu().tolist()
                
                children = []
                for top_k_token_index, token_id in enumerate(top_k_indices):
                    generated_token = tokenizer.convert_ids_to_tokens([token_id])[0].replace('▁', " ")
                    if top_k_probs[top_k_token_index] > 0:
                        new_node = {"name": generated_token, "children": [], "probabilityScore": top_k_probs[top_k_token_index]}
                        children.append(new_node)
                
                current_node_sampling["children"].extend(children)
                next_token_id_val = next_token_id[0][0].cpu().item()
                
                if next_token_id_val in [13, 1]:
                    next_line_counter_sampling += 1
                if next_line_counter_sampling > 2 or next_token_id_val == 13173:
                    should_break_sampling = True
                
                sampling_input_ids = torch.cat([sampling_input_ids, next_token_id], dim=-1)
                generated_token = tokenizer.convert_ids_to_tokens(next_token_id[0])[0].replace('▁', " ")
                current_node_sampling = next(node for node in current_node_sampling["children"] if node["name"] == generated_token)
            
            # Save trees after each step
            save_tree(sampling_tree, os.path.join(OUTDIR, 'g_sampling_tree.json'))
            save_tree(greedy_tree, os.path.join(OUTDIR, 'g_greedy_tree.json'))
            
            greedy_text, sampling_text = update_tree_and_text()
            
            col1, col2 = st.columns(2)
            with col1:
                st.title("Greedy Generation")
                st.markdown(f'<p>{greedy_text}</p>', unsafe_allow_html=True)
                greedy_generated_text = tokenizer.decode(greedy_input_ids[0], skip_special_tokens=False).split("\n")[-1]
                st.markdown(f'<div style="color:green;font-size:24px">{greedy_generated_text}</div>', unsafe_allow_html=True)
            
            with col2:
                st.title("Top P Sampling")
                st.markdown(f'<p>{sampling_text}</p>', unsafe_allow_html=True)
                sampling_generated_text = tokenizer.decode(sampling_input_ids[0], skip_special_tokens=False).split("\n")[-1]
                st.markdown(f'<div style="color:green;font-size:24px">{sampling_generated_text}</div>', unsafe_allow_html=True)
            
            if should_break_greedy and should_break_sampling:
                break

# Streamlit App Configuration

st.set_page_config(layout="wide")

# Load model and tokenizer only once and store in session_state
if "model" not in st.session_state:
    MODEL_NAME = "google/gemma-2b-it"
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    st.session_state.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    st.session_state.model.eval()

tokenizer = st.session_state.tokenizer
model = st.session_state.model

# Input Parameters
col1, col2, col3, col4 = st.columns(4)
with col1:
    input_text = st.text_area("Input Text", value="Write 1 short sentence caption for Dubai\n\n")
with col2:
    num_tokens = st.number_input("Number of Tokens", min_value=1, max_value=100, value=20)
with col3:
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
    st.write(":green[Low Temperature]: Makes peaks sharper. Some tokens will have high confidence, others low.")
    st.write(":red[High Temperature]: Makes peaks less prominent. Smoother distribution. Increases randomness.")
with col4:
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    st.write(":green[Low Top-p]: Model filters only very top probability tokens. Less randomness.")
    st.write(":red[High Top-p]: Model selects from a larger set of tokens. More randomness.")

st.subheader("Model: Gemma 2b")

# Button to generate text
if st.button("Generate"):
    generate_text(model, tokenizer, input_text, num_tokens, temperature, top_p)
