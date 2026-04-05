import streamlit as st
import torch
from PIL import Image
import time
import numpy as np

from src.model import load_model
from src.preprocessing import predict
from src.gradcam import generate_gradcam
from src.lesion_score import calculate_lesion_metrics
from src.explanation import generate_llm_explanation, get_medical_recommendation
from src.report_generator import generate_pdf_report
from src.ui import get_custom_css, render_gauge_chart, render_probabilities_chart

st.set_page_config(page_title="Deep Clinical Insights V4", page_icon="🩺", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

MODEL_PATH = "model.pth"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)
    return model, device

@st.cache_resource
def get_local_llm():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        def generate(prompt):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=120)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("LLM LOADED SUCCESSFULLY")  # debug
        return generate

    except Exception as e:
        st.error(f"Failed to load Transformers: {e}")
        return None

def main():
    st.sidebar.markdown("### AI Engine Configuration")
    use_llm = st.sidebar.checkbox(
        "Enable Local LLM Generator", 
        value=False,
        help="Check this to download & use google/flan-t5-small to translate the mathematical arrays into natural language. If unchecked, defaults to rule-based fallback."
    )
    
    st.markdown('<h1 class="dashboard-title">🥼 Med-AI: Retinal Pathology Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">V4 Hybrid Neuro-Symbolic Pipeline: EfficientNet-B0 + Deterministic Isolation + LLM Translation.</p>', unsafe_allow_html=True)

    model, device = get_model()
    if model is None:
        st.error("Failed to load backend vision model.")
        return
        
    llm_generator = None
    if use_llm:
        with st.sidebar:
             with st.spinner("Loading Local LLM (may take a minute the first time)..."):
                 llm_generator = get_local_llm()

    # Upload section
    uploaded_file = st.file_uploader("Upload Retinal Scan for Analysis (*.png, *.jpg)", type=["png", "jpg", "jpeg"], help="Upload an image to assess probabilities and Grad-CAM++ lesions.")

    if uploaded_file is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error("Could not parse the uploaded image.")
            return

        with st.spinner("Executing Mathematical Calibration Pipeline..."):
            time.sleep(0.5) 
            
            # Inference w/ Temperature Scaling
            pred_class, confidence, probs, image_tensor = predict(model, image, device)
            
            # GradCAM++
            heatmap_overlay, grayscale_cam = generate_gradcam(model, image_tensor, image)
            
            # V4 Structured Quantization Mapping
            lesion_score, affected_pct, binary_mask, json_attrs = calculate_lesion_metrics(grayscale_cam)
            
            # Hybrid LLM Explanation Mapping
            explanation = generate_llm_explanation(json_attrs, CLASS_NAMES[pred_class], llm_generator)
            rec_title, rec_text, rec_color = get_medical_recommendation(pred_class, lesion_score, confidence)
            
            # Generate PDF
            pdf_bytes = generate_pdf_report(
                image, heatmap_overlay, CLASS_NAMES[pred_class], confidence, lesion_score, affected_pct,
                rec_title, rec_text, explanation
            )
        
        # Display Results 
        st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
        
        # TOP BANNER
        st.markdown(f"""
        <div class="stCard" style="border-top: 5px solid {rec_color};" title="Synthesized Medical Inference Recommendation">
            <h2 style="margin: 0;">Predicted Stage: <span style="color: {rec_color}">{CLASS_NAMES[pred_class]}</span></h2>
            <h4 style="margin: 5px 0 0 0; font-weight: 400; color: #555;">Recommendation: <strong>{rec_title}</strong> - {rec_text}</h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown("### Original Scan")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="stCard" title="Grad-CAM++ Focus Localizations">', unsafe_allow_html=True)
            st.markdown("### Spatial Activation Map")
            if heatmap_overlay is not None:
                st.image(heatmap_overlay, use_container_width=True)
            else:
                st.warning("Heatmap parsing failed.")
                
            with st.expander("Show Binary Lesion Mask"):
                if binary_mask is not None:
                      st.image(binary_mask.astype(np.uint8) * 255, use_container_width=True, caption=f"Threshold >= 0.6: {affected_pct:.1f}% retina affected")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            
            meter_c1, meter_c2 = st.columns(2)
            with meter_c1:
                c_text = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
                st.plotly_chart(render_gauge_chart(confidence*100, f"Confidence: {c_text}", "#3b82f6"), use_container_width=True)
            with meter_c2:
                ls_col = "#22c55e" if lesion_score < 30 else "#eab308" if lesion_score < 70 else "#ef4444"
                s_text = "Low" if lesion_score < 30 else "Moderate" if lesion_score < 70 else "High"
                st.plotly_chart(render_gauge_chart(lesion_score, f"Lesion: {s_text}", ls_col), use_container_width=True)

            st.markdown("### Pathway Probabilities")
            st.plotly_chart(render_probabilities_chart(probs, CLASS_NAMES), use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown("### 🧠 Autonomous Clinical Interpretation")
        st.markdown(f'<div class="medical-insight-box">{explanation}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📁 Download Certified Clinical PDF Report",
            data=bytes(pdf_bytes),
            file_name=f"DR_Report_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Generates an offline clinical brief via fpdf2 using current pipeline metrics."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True) 

if __name__ == "__main__":
    main()
