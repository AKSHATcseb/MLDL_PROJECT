"""
Glaucoma Detection Research Dashboard: Streamlit Application.
Multi-modal diagnostic assistance integrating:
1. ResNet34 U-Net Segmentation (Optic Disc & Cup)
2. Biomarker Extraction (vCDR, Area CDR, Rim Area)
3. EfficientNet-B0 Classification
4. Grad-CAM Explainable AI Heatmap & Overlay
"""

import os
import sys
import io
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.integration.inference import GlaucomaInferenceEngine

st.set_page_config(
    page_title="Glaucoma AI - Clinical Research Prototype",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0A3D62;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #3C6382;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFA000;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #856404;
        font-size: 0.92rem;
    }
    .metric-card {
        background: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E3799;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6C757D;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    try:
        return GlaucomaInferenceEngine()
    except Exception as e:
        st.error(f"Error loading trained models: {e}")
        return None

def main():
    st.markdown('<div class="main-title">Glaucoma Screening & Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Modal Deep Learning: Segmentation • Classification • Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Research Disclaimer:</strong> This application is an AI-based research prototype. 
        It is not a substitute for professional ophthalmological examination, visual field testing, or medical diagnosis.
    </div>
    """, unsafe_allow_html=True)
    
    engine = load_engine()
    if engine is None:
        st.stop()
        
    # Sidebar
    st.sidebar.header("📋 Controls & Samples")
    preset_opt = st.sidebar.selectbox(
        "Load Preset Sample:",
        ["None (Custom Upload)", "ACRIMA Glaucoma Case (Im322)", "ACRIMA Normal Case (Im013)", "REFUGE Glaucoma Case (g0001)", "REFUGE Normal Case (n0001)"]
    )
    
    uploaded_file = st.sidebar.file_uploader("Or Upload Retinal Fundus Photo:", type=["jpg", "jpeg", "png"])
    
    input_img = None
    sample_id = "Uploaded Image"
    
    # Preset sample mappings
    preset_paths = {
        "ACRIMA Glaucoma Case (Im322)": os.path.join(PROJECT_ROOT, "dataset", "test", "Glaucoma", "Im322_g_ACRIMA.jpg"),
        "ACRIMA Normal Case (Im013)": os.path.join(PROJECT_ROOT, "dataset", "test", "Non Glaucoma", "Im013_ACRIMA.jpg"),
        "REFUGE Glaucoma Case (g0001)": os.path.join(PROJECT_ROOT, "dataset", "REFUGE2", "train", "images", "g0001.jpg"),
        "REFUGE Normal Case (n0001)": os.path.join(PROJECT_ROOT, "dataset", "REFUGE2", "train", "images", "n0001.jpg")
    }
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, 1)
        input_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sample_id = uploaded_file.name
    elif preset_opt in preset_paths and os.path.exists(preset_paths[preset_opt]):
        p = preset_paths[preset_opt]
        input_img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        sample_id = os.path.basename(p)
        
    if input_img is None:
        st.info("👈 Please select a preset test sample from the sidebar or upload a fundus photograph to begin analysis.")
        return
        
    col_l, col_r = st.columns([1, 1.2])
    with col_l:
        st.subheader("Original Fundus Photograph")
        st.image(input_img, caption=f"Active Sample: {sample_id} ({input_img.shape[1]}x{input_img.shape[0]} px)", use_container_width=True)
        run_btn = st.button("🚀 Run Multi-Modal AI Analysis", type="primary", use_container_width=True)
        
    if run_btn:
        with st.spinner("Analyzing fundus morphology, segmenting Optic Disc & Cup, and computing Grad-CAM..."):
            res = engine.predict(input_img)
            
        prob = res['glaucoma_probability']
        pred_label = res['prediction_label']
        thresh = res['operating_threshold']
        feats = res['clinical_features']
        
        with col_r:
            st.subheader("Diagnostic Assessment")
            is_glaucoma = (pred_label == "Glaucoma")
            badge_color = "#D63031" if is_glaucoma else "#00B894"
            st.markdown(f"""
            <div style="background-color: {badge_color}15; border: 2px solid {badge_color}; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h3 style="color: {badge_color}; margin:0;">AI Prediction: {pred_label}</h3>
                <p style="margin: 5px 0 0 0; color: #2D3436;">
                    Confidence Score: <strong>{prob * 100:.2f}%</strong> | Operating Decision Threshold: {thresh:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(prob))
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{feats["vcdr"]:.3f}</div><div class="metric-label">Vertical CDR (vCDR)</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{feats["area_cdr"]:.3f}</div><div class="metric-label">Cup/Disc Area Ratio</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{feats["rim_area_px"]:,} px</div><div class="metric-label">Neuroretinal Rim Area</div></div>', unsafe_allow_html=True)
                
            st.markdown(f"**Clinical vCDR Heuristic:** `{feats['clinical_vcdr_risk']}`")
            
        st.divider()
        st.subheader("🔬 Detailed Visual Outputs")
        
        tab1, tab2, tab3 = st.tabs(["1. Semantic Segmentation", "2. Explainable AI (Grad-CAM)", "3. Raw Masks & Diameters"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(input_img, caption="Input Fundus Image", use_container_width=True)
            with c2:
                st.image(res['mask_overlay'], caption="Segmentation Overlay\n(Cyan: Disc Rim, Red: Cup)", use_container_width=True)
            with c3:
                # Color coded mask
                cm = np.zeros_like(input_img)
                cm[res['predicted_mask'] == 1] = [0, 200, 255]
                cm[res['predicted_mask'] == 2] = [255, 50, 50]
                st.image(cm, caption="Discrete 3-Class Segmentation Mask", use_container_width=True)
                
        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(input_img, caption="Input Fundus Image", use_container_width=True)
            with c2:
                st.image(res['gradcam_heatmap'], caption="Grad-CAM Activation Heatmap", use_container_width=True)
            with c3:
                st.image(res['gradcam_overlay'], caption="Grad-CAM Heatmap Overlay", use_container_width=True)
            st.caption("ℹ️ Grad-CAM visualizes gradients flowing into the final convolutional layer of EfficientNet-B0 to confirm sensitivity to the optic nerve head.")
            
        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.image(res['optic_disc_mask'], caption=f"Optic Disc Binary Mask (V-Diam: {feats['disc_vertical_diameter_px']} px)", use_container_width=True)
            with c2:
                st.image(res['optic_cup_mask'], caption=f"Optic Cup Binary Mask (V-Diam: {feats['cup_vertical_diameter_px']} px)", use_container_width=True)
                
        # Download report
        report_data = {
            'sample_id': sample_id,
            'prediction': pred_label,
            'glaucoma_probability': prob,
            'operating_threshold': thresh,
            'clinical_features': feats,
            'disclaimer': res['disclaimer']
        }
        st.download_button(
            label="💾 Download Comprehensive JSON Analysis Report",
            data=json.dumps(report_data, indent=2),
            file_name=f"glaucoma_report_{sample_id}.json",
            mime="application/json"
        )

if __name__ == '__main__':
    main()
