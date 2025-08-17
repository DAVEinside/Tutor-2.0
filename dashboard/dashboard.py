import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
from PIL import Image
import io

st.set_page_config(
    page_title="Math Tutor Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Math Tutor - Evaluation Dashboard")
st.markdown("Internal tool for evaluating OCR and LLM performance")

# Initialize session state
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []

# Sidebar for controls
st.sidebar.header("Controls")

# File uploader for test images
uploaded_file = st.sidebar.file_uploader(
    "Upload test image", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload a handwritten math problem for testing"
)

# Backend URL configuration
backend_url = st.sidebar.text_input(
    "Backend URL", 
    value="http://localhost:8000",
    help="URL of the FastAPI backend"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Image Analysis")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Prepare file for upload
                    files = {
                        'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    
                    # Call OCR endpoint
                    response = requests.post(f"{backend_url}/upload-frame", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('success'):
                            st.success("✅ OCR Analysis Complete")
                            
                            # Display OCR results
                            st.subheader("OCR Results")
                            st.write("**Raw OCR Output:**")
                            st.code(result.get('extracted_text', 'No text extracted'))
                            
                            st.write("**Cleaned Text:**")
                            st.code(result.get('cleaned_text', 'No cleaned text'))
                            
                            # Store for evaluation
                            st.session_state.current_ocr = result
                            
                        else:
                            st.error(f"❌ OCR Failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ Request failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    st.header("🧠 LLM Evaluation")
    
    # Manual text input option
    manual_text = st.text_area(
        "Or enter text manually:",
        help="Enter mathematical text for evaluation",
        height=100
    )
    
    # Evaluate button
    if st.button("📝 Evaluate Solution", type="secondary"):
        text_to_evaluate = None
        
        if manual_text.strip():
            text_to_evaluate = manual_text.strip()
        elif hasattr(st.session_state, 'current_ocr'):
            text_to_evaluate = st.session_state.current_ocr.get('cleaned_text', '')
        
        if text_to_evaluate:
            with st.spinner("Evaluating solution..."):
                try:
                    # Call evaluation endpoint
                    eval_data = {"text": text_to_evaluate}
                    response = requests.post(
                        f"{backend_url}/evaluate", 
                        json=eval_data,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Evaluation Complete")
                        
                        # Display evaluation results
                        st.subheader("Evaluation Results")
                        st.markdown(result.get('evaluation', 'No evaluation available'), unsafe_allow_html=True)
                        
                        # Store evaluation for review
                        evaluation_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'input_text': text_to_evaluate,
                            'evaluation': result.get('evaluation', ''),
                            'detailed_result': result.get('detailed_result', {}),
                            'image_filename': uploaded_file.name if uploaded_file else 'Manual Input'
                        }
                        st.session_state.evaluations.append(evaluation_entry)
                        
                    else:
                        st.error(f"❌ Evaluation failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please provide text to evaluate (either upload an image or enter text manually)")

# Evaluation History
if st.session_state.evaluations:
    st.header("📋 Evaluation History")
    
    # Convert to DataFrame for better display
    df_data = []
    for i, eval_entry in enumerate(st.session_state.evaluations):
        detailed = eval_entry.get('detailed_result', {})
        df_data.append({
            'ID': i + 1,
            'Timestamp': eval_entry['timestamp'][:19],  # Remove microseconds
            'Input': eval_entry['input_text'][:50] + '...' if len(eval_entry['input_text']) > 50 else eval_entry['input_text'],
            'Correct': detailed.get('is_correct', 'Unknown'),
            'Source': eval_entry['image_filename']
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Detailed view selector
    selected_eval = st.selectbox(
        "Select evaluation for detailed view:",
        options=range(len(st.session_state.evaluations)),
        format_func=lambda x: f"Evaluation #{x+1} - {st.session_state.evaluations[x]['timestamp'][:19]}"
    )
    
    if selected_eval is not None:
        eval_detail = st.session_state.evaluations[selected_eval]
        
        with st.expander("📊 Detailed Evaluation View", expanded=True):
            col_detail1, col_detail2 = st.columns([1, 1])
            
            with col_detail1:
                st.write("**Input Text:**")
                st.code(eval_detail['input_text'])
                
                st.write("**Source:**")
                st.write(eval_detail['image_filename'])
            
            with col_detail2:
                st.write("**Evaluation Result:**")
                st.markdown(eval_detail['evaluation'], unsafe_allow_html=True)
                
                if eval_detail.get('detailed_result'):
                    st.write("**Raw Result:**")
                    st.json(eval_detail['detailed_result'])
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.evaluations = []
        st.experimental_rerun()

# Footer with system info
st.markdown("---")
st.markdown("**System Status**")
try:
    health_response = requests.get(f"{backend_url}/", timeout=5)
    if health_response.status_code == 200:
        st.success(f"✅ Backend connected ({backend_url})")
    else:
        st.error(f"❌ Backend error: {health_response.status_code}")
except:
    st.error(f"❌ Backend unreachable ({backend_url})")