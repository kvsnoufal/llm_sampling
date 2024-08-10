import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import torch.nn.functional as F
import json
import os

def get_level_names_greedy(node):
    """Recursively collect names of all nodes grouped by their levels, highlighting the first child and root node."""
    levels = {}
    
    def traverse(node, level, is_first_child=False, is_root=False):
        if level not in levels:
            levels[level] = []
        if is_root:
            levels[level].append(f"<span style='color:blue;'>{node['name']} </span>")
        elif is_first_child:
            levels[level].append(f"<span style='color:blue;'>{node['name']}</span><span style='color:blue;font-size:8px;font-style:strong;vertical-align: sub;'> {int(node['probabilityScore']*100)}%</span>")
        else:
            levels[level].append(f"<span style='color:grey;'>{node['name']}</span><span style='color:grey;font-size:8px;font-style:strong;vertical-align: sub;'> {int(node['probabilityScore']*100)}%</span>")
        
        if 'children' in node and node['children']:
            for idx, child in enumerate(node['children']):
                traverse(child, level + 1, is_first_child=(idx == 0))
    
    traverse(node, 0, is_root=True)
    return levels

def get_level_names_sampling(node):
    """Recursively collect names of all nodes grouped by their levels, highlighting the first child and root node."""
    levels = {}
    
    def traverse(node, level, has_children=False, is_root=False,child_index=0,parent_index=0):
        # global global_top_indentation
        if level not in levels:
            levels[level] = []

        if is_root:
            levels[level].append(f"<span style='color:blue;'>{node['name']} </span>")
        elif has_children:
            levels[level].append(f"<span style='color:blue;'>{node['name']}</span><span style='color:blue;font-size:8px;font-style:strong;vertical-align: sub;'> {int(node['probabilityScore']*100)}%</span>")
        else:
            levels[level].append(f"<span style='color:grey;'>{node['name']}</span><span style='color:grey;font-size:8px;font-style:strong;vertical-align: sub;'> {int(node['probabilityScore']*100)}%</span>")
        parent_index = child_index
        if 'children' in node and node['children']:
            for idx, child in enumerate(node['children']):
                
                traverse(child, level + 1, has_children=(len(child['children'])>0),child_index=idx,parent_index=parent_index)
    
    traverse(node, 0, is_root=True)
    return levels
if "model" not in st.session_state:
    MODEL_NAME = "google/gemma-2b-it"
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    st.session_state.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    st.session_state.model.eval()

tokenizer = st.session_state.tokenizer
model = st.session_state.model
st.set_page_config(layout="wide")

col1, col2, col3, col4 = st.columns(4)

with col1:
    input_text = st.text_area("Input Text", value="Write 1 short sentence caption for Dubai\n\n")
    
with col2:
    num_tokens = st.number_input("Number of Tokens", min_value=1, max_value=100, value=20)
    
with col3:
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.4, step=0.1)
    st.write(":green[Low Temperature]: makes peaks sharper. Some tokens will have very high confidence, but others will be very low confidence")
    st.write(":red[High Temperature]: makes peaks less prominent. Smoother distribution. Increases randomness")
    
with col4:
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    st.write(":green[Low Top-p]: Model filters only very top probably tokens before sampling. Less randomness")
    st.write(":red[High Top-p]: Model selects from larger set of tokens. More randomness")
st.subheader("Model : Gemma 2b")

# Button to generate text
if st.button("Generate"):

    col1, col2 = st.columns(2)

    with col1:
        st.title("Greedy Generation")
        greedy_completed_text_placeholder = st.empty()
        greedy_text_placeholder = st.empty()

    with col2:
        st.title("Top P Sampling")
        sampling_completed_text_placeholder = st.empty()
        sampling_text_placeholder = st.empty()

    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    # Greedy and sampling initialization
    greedy_input_ids = input_ids.input_ids.clone()
    sampling_input_ids = input_ids.input_ids.clone()

    next_line_counter_greedy = 2
    next_line_counter_sampling = 2

    greedy_tree = {"name": "<bos>", "children": []}
    sampling_tree = {"name": "<bos>", "children": []}

    current_node_greedy = greedy_tree
    current_node_sampling = sampling_tree

    OUTDIR = "output_v2"
    os.makedirs(OUTDIR, exist_ok=True)

    def save_trees():
        with open(OUTDIR+'/g_sampling_tree.json', 'w') as f:
            json.dump(sampling_tree, f, indent=2)
        with open(OUTDIR+'/g_greedy_tree.json', 'w') as f:
            json.dump(greedy_tree, f, indent=2)

    should_break_greedy = False
    should_break_sampling = False

    with torch.no_grad():
        for _ in range(num_tokens):
            # Greedy approach step
            if not should_break_greedy:
                outputs = model(greedy_input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(logits, 10, dim=-1)
                top_k_probs = F.softmax(top_k_logits, dim=-1)[0].cpu().tolist()
                top_k_indices = top_k_indices[0].cpu().tolist()

                children = []
                for top_k_token_index, token_id in enumerate(top_k_indices):
                    if top_k_token_index == 0:
                        if token_id == 1:
                            next_line_counter_greedy += 1
                        if next_line_counter_greedy > 2 or token_id == 13173:
                            should_break_greedy = True

                    generated_token = tokenizer.convert_ids_to_tokens([token_id])[0].replace('▁'," ")     
                    new_node = {
                        "name": generated_token,
                        "children": [],
                        "probabilityScore": top_k_probs[top_k_token_index]
                    }
                    children.append(new_node)

                current_node_greedy["children"].extend(children)
                if children:
                    current_node_greedy = children[0]

                # st.write(f"Greedy Tree after step {_+1}:")
                # st.write(str(current_node_greedy))

                if not should_break_greedy:
                    first_token_tensor = torch.tensor([top_k_indices[0]], device=greedy_input_ids.device).unsqueeze(-1)
                    greedy_input_ids = torch.cat([greedy_input_ids, first_token_tensor], dim=-1)

            # Sampling approach step
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
                    generated_token = tokenizer.convert_ids_to_tokens([token_id])[0].replace('▁'," ")
                    if top_k_probs[top_k_token_index] > 0:
                        new_node = {
                            "name": generated_token,
                            "children": [],
                            "probabilityScore": top_k_probs[top_k_token_index]
                        }
                        children.append(new_node)

                current_node_sampling["children"].extend(children)

                # st.write(f"Sampling Tree after step {_+1}:")
                # st.write(str(current_node_sampling))

                if next_token_id[0][0].cpu().item() == 13:
                    next_line_counter_sampling += 1
                if next_token_id[0][0].cpu().item() == 1:
                    next_line_counter_sampling += 1
                if next_line_counter_sampling > 2 or next_token_id[0][0].cpu().item() == 13173:
                    should_break_sampling = True

                sampling_input_ids = torch.cat([sampling_input_ids, next_token_id], dim=-1)

                generated_token = tokenizer.convert_ids_to_tokens(next_token_id[0])[0].replace('▁'," ")
                current_node_sampling = next(node for node in current_node_sampling["children"] if node["name"] == generated_token)

            # Save trees after each step
            save_trees()

            greedy_levels = get_level_names_greedy(greedy_tree)
            level_html = ""
            for level in range(len(greedy_levels)):
                if level in greedy_levels:
                    # Combine nodes at the same level into a vertical stack
                    level_text = "".join(
                        f"<div style='padding:5px 0; border-bottom:1px solid grey; font-size:12px;'>{node}</div>"
                        for node in greedy_levels[level]
                    )
                    level_html += f"<div style='display:flex; flex-direction:column; align-items:flex-start; margin-right:2px; max-width:200px;'>{level_text}</div>"

            # Wrap each level's content in a separate div and align them horizontally
            text_to_display = f"<div style='display:flex; flex-direction:row; overflow-x:auto;'>{level_html}</div>"

            # Update the text in the placeholder
            greedy_text_placeholder.markdown(f'<p>{text_to_display}</p>', unsafe_allow_html=True)


            sampling_levels = get_level_names_sampling(sampling_tree)

            level_html = ""
            for level in range(len(sampling_levels)):
                if level in sampling_levels:
                    # Combine nodes at the same level into a vertical stack
                    level_text = "".join(
                        f"<div style='padding:5px 0; border-bottom:1px solid grey; font-size:12px;'>{node}</div>"
                        for node in sampling_levels[level]
                    )
                    level_html += f"<div style='display:flex; flex-direction:column; align-items:flex-start; margin-right:2px; max-width:200px;'>{level_text}</div>"

            # Wrap each level's content in a separate div and align them horizontally
            text_to_display = f"<div style='display:flex; flex-direction:row; overflow-x:auto;'>{level_html}</div>"

            # Update the text in the placeholder
            sampling_text_placeholder.markdown(f'<p>{text_to_display}</p>', unsafe_allow_html=True)

            greedy_generated_text = tokenizer.decode(greedy_input_ids[0], skip_special_tokens=False).split("\n")[-1]
            sampling_generated_text = tokenizer.decode(sampling_input_ids[0], skip_special_tokens=False).split("\n")[-1]
            greedy_completed_text_placeholder.markdown(f'<div style="color:green;font-size:24px"><p style="color:green;font-size:24px">{greedy_generated_text}</p></div>', unsafe_allow_html=True)
            sampling_completed_text_placeholder.markdown(f'<div style="color:green;font-size:24px"><p style="color:green;font-size:24px">{sampling_generated_text}</p></div>', unsafe_allow_html=True)
            # Break if both have finished
            if should_break_greedy and should_break_sampling:
                # st.write("Stopping criteria met for both")
                break

