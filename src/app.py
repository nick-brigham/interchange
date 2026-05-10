import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = 'models/merged'
FINETUNED_DIR = 'models/merged'

st.set_page_config(
    page_title="Part Interchange Assistant",
    page_icon="🔧",
    layout="centered"
)

@st.cache_resource
def load_model():
    with st.spinner("Loading model... this takes a minute on first run."):
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map='cpu'
        )
        model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
        model.eval()
    return model, tokenizer

def predict(description, model, tokenizer):
    prompt = f"""You are a parts interchange assistant. Identify the correct part number.

Customer description: {description}
Part number:"""

    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if 'Part number:' in full_output:
        answer = full_output.split('Part number:')[-1].strip()
        answer = answer.split('\n')[0].strip()
    else:
        answer = full_output.strip()

    return answer

st.title("🔧 Part Interchange Assistant")
st.markdown("Enter a customer part description to identify the correct manufacturer part number.")

model, tokenizer = load_model()

st.success("Model loaded and ready.")

with st.form("interchange_form"):
    description = st.text_input(
        "Customer Description",
        placeholder="e.g. 5VX840 SUPER HC V-BELT"
    )
    submitted = st.form_submit_button("Find Part Number")

if submitted:
    if not description.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("Looking up part number..."):
            result = predict(description, model, tokenizer)
        
        st.markdown("---")
        st.markdown("### Result")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input Description**")
            st.info(description)
        with col2:
            st.markdown("**Predicted Part Number**")
            st.success(result)

st.markdown("---")
st.markdown(
    "Built with Llama 3.2 1B · Fine-tuned with LoRA · "
    "Pipeline: pandas → scikit-learn → HuggingFace Transformers"
)