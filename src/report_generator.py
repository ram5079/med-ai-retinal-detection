import os
import tempfile
from datetime import datetime
from fpdf import FPDF
from PIL import Image

class ClinicalReportPDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font("Arial", "B", 18)
        self.set_text_color(30, 58, 138) # Dark blue
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, "Diabetic Retinopathy Clinical Report", align="C")
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font("Arial", "I", 8)
        self.set_text_color(128)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, f"Generated automatically by AI System  |  {timestamp}  |  Page {self.page_no()}", align="C")

def generate_pdf_report(original_img, heatmap_img, diag_class, confidence, lesion_score, affected_pct, 
                        rec_title, rec_text, explanation):
    """
    Generates a PDF report containing the clinical findings.
    original_img and heatmap_img should be PIL Images.
    Returns the PDF as bytes.
    """
    pdf = ClinicalReportPDF()
    pdf.add_page()
    
    # Save images to temp files so FPDF can read them
    with tempfile.TemporaryDirectory() as tmpdirname:
        orig_path = os.path.join(tmpdirname, "original.jpg")
        heat_path = os.path.join(tmpdirname, "heatmap.jpg")
        
        # Ensure RGB
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        original_img.save(orig_path)
        
        if heatmap_img is not None:
            # heatmap_img is a numpy array in UI, so we need to convert to PIL first
            import numpy as np
            if isinstance(heatmap_img, np.ndarray):
                # Ensure it's 255 scaled
                if heatmap_img.dtype == np.float32 or heatmap_img.dtype == np.float64:
                    h_img = Image.fromarray((heatmap_img * 255).astype(np.uint8))
                else:
                    h_img = Image.fromarray(heatmap_img)
            else:
                h_img = heatmap_img
                
            if h_img.mode != 'RGB':
                 h_img = h_img.convert('RGB')
            h_img.save(heat_path)

        # Section: AI Diagnosis Overview
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "AI Diagnosis Overview", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(50, 8, "Predicted Stage:", border=0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, str(diag_class), ln=True)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(50, 8, "Model Confidence:", border=0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"{confidence*100:.1f}%", ln=True)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(50, 8, "Lesion Severity Score:", border=0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"{lesion_score}/100", ln=True)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(50, 8, "Affected Area:", border=0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"{affected_pct:.1f}%", ln=True)
        
        pdf.ln(10)
        
        # Section: Imaging
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Retinal Imaging & AI Mapping", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Calculate positions to center side-by-side images
        img_w = 80
        x_orig = 20
        x_heat = 110
        y_curr = pdf.get_y()
        
        pdf.image(orig_path, x=x_orig, y=y_curr, w=img_w)
        if heatmap_img is not None:
            pdf.image(heat_path, x=x_heat, y=y_curr, w=img_w)
            
        pdf.set_y(y_curr + img_w + 5)
        pdf.set_font("Arial", "I", 10)
        pdf.text(x_orig + 10, pdf.get_y(), "Original Input Image")
        if heatmap_img is not None:
            pdf.text(x_heat + 10, pdf.get_y(), "Grad-CAM Activation Map")
            
        pdf.ln(15)
        
        # Section: Explanation 
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Clinical AI Insight & Rationale", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Arial", "", 11)
        # clean explanation text
        clean_exp = explanation.replace("**", "").replace("\\n", " ")
        pdf.multi_cell(0, 6, clean_exp)
        pdf.ln(10)
        
        # Section: Recommendation
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Medical Recommendation", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, rec_title, ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, rec_text)
        
        return pdf.output(dest="S") # Returns a byte string
