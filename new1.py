import streamlit as st
import pdfplumber
import spacy
import re
from io import BytesIO
import pandas as pd
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import cv2
import numpy as np
import os

# Sab se pehla Streamlit command
st.set_page_config(page_title="AI Medical Report & MCQ Analyzer", layout="wide")

# Tesseract ka path set karo
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Agar Tesseract kisi aur jagah install hai, toh yahan path update karo

# SciSpacy model load karo
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    st.error("SciSpacy model install karo: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
    st.stop()

# Tesseract check karo
try:
    tesseract_version = pytesseract.get_tesseract_version()
    st.sidebar.info(f"Tesseract OCR version {tesseract_version} mil gaya.")
except Exception as e:
    st.error("Tesseract OCR install nahi hai ya PATH mein nahi hai.")
    st.error("1. Tesseract download aur install karo: https://github.com/UB-Mannheim/tesseract/wiki")
    st.error("2. Tesseract ko PATH mein add karo ya code mein path update karo (line 22).")
    st.error("3. Default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    st.error(f"Error: {str(e)}")
    st.stop()

# Image preprocess karne ka function
def preprocess_image(image):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_img = Image.fromarray(binary)
        return processed_img
    except Exception as e:
        st.warning(f"Image preprocess nahi hua: {str(e)}. Original image use hoga.")
        return image

# PDF ya image se text nikalne ka function
def extract_text(file, file_type):
    try:
        text = ""
        if file_type == "pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        st.warning("PDF page scanned image hai. OCR try kar raha...")
                        img = page.to_image().original
                        processed_img = preprocess_image(img)
                        ocr_text = pytesseract.image_to_string(processed_img)
                        if ocr_text.strip():
                            text += ocr_text + "\n"
                        else:
                            st.warning("Is PDF page se text nahi nikla.")
                            text += "No text extracted from this page.\n"
        else:  # Image (JPEG/PNG)
            img = Image.open(file)
            processed_img = preprocess_image(img)
            ocr_text = pytesseract.image_to_string(processed_img)
            if ocr_text.strip():
                text = ocr_text
            else:
                st.error("Image se text nahi nikla. Image high-resolution aur clear text wali honi chahiye.")
                text = "Error: No text extracted from image."
        
        if not text.strip() or text.strip().startswith("Error") or text.strip().startswith("No text extracted"):
            return f"Error: {file_type} se text nahi nikla. Doosra file try karo ya text paste karo."
        return text
    except Exception as e:
        st.error(f"{file_type} process mein error: {str(e)}")
        return f"Error processing {file_type}: {str(e)}"

# MCQ answers nikalne ka function
def extract_mcq_answers(image_file):
    try:
        img = Image.open(image_file)
        processed_img = preprocess_image(img)
        ocr_text = pytesseract.image_to_string(processed_img)
        
        # Question number aur answer detect karo (jaise "1. B")
        mcq_pattern = r"(\d+)\.\s*([A-E])"
        matches = re.findall(mcq_pattern, ocr_text, re.IGNORECASE)
        
        if not matches:
            return "Error: MCQ answers nahi mile. Image mein clear question numbers aur answers hone chahiye (jaise '1. B')."
        
        answers = {f"Q{int(q)}": ans for q, ans in matches}
        return answers
    except Exception as e:
        return f"Error processing MCQ image: {str(e)}"

# Lab values categorize karne ka function
def categorize_value(value, ref_range):
    try:
        value = float(value)
        if "-" in ref_range:
            low, high = map(float, ref_range.split("-"))
            if value < low or value > high:
                if value < low * 0.8 or value > high * 1.2:
                    return "Critical"
                return "Borderline"
            return "Normal"
        elif ref_range.startswith("<"):
            max_val = float(ref_range[1:])
            if value > max_val:
                if value > max_val * 1.2:
                    return "Critical"
                return "Borderline"
            return "Normal"
        return "Normal"
    except:
        return "Unknown"

# AI explanation banane ka function
def generate_explanation(test_name, value, unit, ref_range, status):
    prompt = f"Samjhao ke agar patient ka {test_name} {value} {unit} hai, aur normal range {ref_range} hai, toh iska kya matlab hai."
    if status != "Normal":
        explanation = f"Aapka {test_name} level {value} {unit} hai jo {status.lower()} hai. "
        if "SGPT (ALT)" in test_name or "SGOT (AST)" in test_name:
            explanation += "Iska matlab hai ke aapka liver par stress ya damage ho sakta hai. Iski wajah alcohol, dawaiyan, ya hepatitis jaisi bimari ho sakti hai."
        elif "BILIRUBIN" in test_name:
            explanation += "Yeh liver ya gallbladder mein masla dikha sakta hai, jaise jaundice ya blockage."
        explanation += " Doctor se detail check-up karwao."
    else:
        explanation = f"Aapka {test_name} level {value} {unit} normal hai, yani yeh hissa bilkul theek hai!"
    return explanation

# Risk summary aur suggestions banane ka function
def generate_risk_summary(lab_data):
    critical_count = sum(1 for item in lab_data if item["Status"] == "Critical")
    borderline_count = sum(1 for item in lab_data if item["Status"] == "Borderline")
    
    summary = "Aapki medical report ka analysis dikha raha hai: "
    if critical_count > 0:
        summary += f"{critical_count} critical value(s) jo turant dhyan mangte hain. "
    if borderline_count > 0:
        summary += f"{borderline_count} borderline value(s) jo monitor karne chahiye. "
    if critical_count == 0 and borderline_count == 0:
        summary += "Sab values normal range mein hain. Apni sehat ka khayal rakho!"
    
    suggestions = []
    for item in lab_data:
        if item["Status"] == "Critical":
            if "SGPT (ALT)" in item["Test"] or "SGOT (AST)" in item["Test"]:
                suggestions.append("Turant hepatologist se liver check-up karwao.")
            elif "BILIRUBIN" in item["Test"]:
                suggestions.append("Liver ya gallbladder ke masle ke liye doctor se milo.")
        elif item["Status"] == "Borderline":
            suggestions.append(f"{item['Test']} ko monitor karo aur doctor se baat karo.")
    
    return summary, suggestions

# Medical report analyze karne ka function
def analyze_medical_report(text):
    doc = nlp(text)
    
    diagnoses = []
    lab_values = []
    medications = []
    abnormal_values = []
    insights = []
    explanations = []
    
    lab_pattern = r"(\S.*\S)\s+([\d.]+)\s+(mg/dL|U/L|U/I|mmol/L|%|mmHg)\s+([\d.<>-]+)"
    lab_matches = re.findall(lab_pattern, text, re.IGNORECASE)
    
    lab_data = []
    for match in lab_matches:
        test_name, value, unit, ref_range = match
        lab_values.append(f"{test_name}: {value} {unit}")
        
        try:
            value_float = float(value)
            status = categorize_value(value, ref_range)
            lab_data.append({
                "Test": test_name,
                "Result": f"{value} {unit}",
                "Reference Range": ref_range,
                "Status": status
            })
            
            if status in ["Critical", "Borderline"]:
                abnormal_values.append(f"{test_name}: {value} {unit} ({status}, Outside range: {ref_range})")
            
            explanation = generate_explanation(test_name, value, unit, ref_range, status)
            explanations.append({"Test": test_name, "Explanation": explanation})
        except:
            continue
    
    for abnormal in abnormal_values:
        if "SGPT (ALT)" in abnormal or "SGOT (AST)" in abnormal:
            insights.append(f"{abnormal}: Zyada ALT/AST liver ke stress ya damage ko dikha sakta hai. Hepatologist se milo.")
        elif "BILIRUBIN" in abnormal:
            insights.append(f"{abnormal}: Abnormal bilirubin liver ya gallbladder ke masle dikha sakta hai. Tests karwao.")
    
    diagnosis_pattern = r"(Diagnosis|Impression|Finding):\s*([^\n]+)"
    diagnosis_matches = re.findall(diagnosis_pattern, text, re.IGNORECASE)
    for match in diagnosis_matches:
        diagnoses.append(match[1].strip())
    
    medication_pattern = r"(Medication|Rx|Prescription):\s*([^\n]+)"
    medication_matches = re.findall(medication_pattern, text, re.IGNORECASE)
    for match in medication_matches:
        medications.append(match[1].strip())
    
    diagnoses = list(set(diagnoses))
    lab_values = list(set(lab_values))
    medications = list(set(medications))
    
    risk_summary, suggestions = generate_risk_summary(lab_data)
    
    summary = {
        "Diagnoses": diagnoses if diagnoses else ["Koi diagnosis nahi mili"],
        "Lab Values": lab_values,
        "Medications": medications if medications else ["Koi dawaiyan nahi likhi gayi"],
        "Abnormal Findings": abnormal_values if abnormal_values else ["Koi abnormal findings nahi"],
        "Lab Data": lab_data,
        "Insights": insights if insights else ["Koi khas insight nahi"],
        "Explanations": explanations,
        "Risk Summary": risk_summary,
        "Suggestions": suggestions if suggestions else ["Koi khas follow-up nahi chahiye."]
    }
    
    return summary, text

# Bar chart banane ka function
def create_bar_chart(lab_data):
    tests = [item["Test"] for item in lab_data]
    results = [float(re.search(r"[\d.]+", item["Result"]).group()) for item in lab_data]
    
    colors = []
    for item in lab_data:
        if item["Status"] == "Critical":
            colors.append("#ff3333")
        elif item["Status"] == "Borderline":
            colors.append("#ffcc00")
        else:
            colors.append("#66b3ff")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tests, results, color=colors)
    ax.set_xlabel("Tests")
    ax.set_ylabel("Results")
    ax.set_title("Lab Results Visualization")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_str}"

# PDF report banane ka function
def generate_pdf_report(summary):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Medical Report Analysis", styles['Title']))
    
    elements.append(Paragraph("Risk Summary:", styles['Heading2']))
    elements.append(Paragraph(summary["Risk Summary"], styles['Normal']))
    
    if summary["Lab Data"]:
        data = [["Test", "Result", "Reference Range", "Status"]]
        for item in summary["Lab Data"]:
            data.append([item["Test"], item["Result"], item["Reference Range"], item["Status"]])
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#2c3e50'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#f0f2f6'),
            ('GRID', (0, 0), (-1, -1), 1, '#000000'),
        ]))
        elements.append(table)
    
    elements.append(Paragraph("Diagnoses Identified:", styles['Heading2']))
    for diag in summary["Diagnoses"]:
        elements.append(Paragraph(f"- {diag}", styles['Normal']))
    
    elements.append(Paragraph("Medications:", styles['Heading2']))
    for med in summary["Medications"]:
        elements.append(Paragraph(f"- {med}", styles['Normal']))
    
    elements.append(Paragraph("Abnormal Findings:", styles['Heading2']))
    for finding in summary["Abnormal Findings"]:
        elements.append(Paragraph(f"- {finding}", styles['Normal']))
    
    elements.append(Paragraph("Explanations:", styles['Heading2']))
    for exp in summary["Explanations"]:
        elements.append(Paragraph(f"{exp['Test']}: {exp['Explanation']}", styles['Normal']))
    
    elements.append(Paragraph("Follow-up Suggestions:", styles['Heading2']))
    for suggestion in summary["Suggestions"]:
        elements.append(Paragraph(f"- {suggestion}", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit Interface
def main():
    # Custom CSS for modern UI
    st.markdown("""
        <style>
        .main {background-color: #e6f3ff;}
        .title {color: #2c3e50; font-size: 40px; font-weight: bold; text-align: center; margin-bottom: 10px;}
        .subheader {color: #34495e; font-size: 24px; font-weight: 500;}
        .result-box {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 20px;}
        .raw-text {background-color: #f8f9fa; padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre; max-height: 400px; overflow-y: auto; border: 1px solid #ddd;}
        .sidebar .sidebar-content {background-color: #2c3e50; color: #ffffff;}
        .stButton>button {background-color: #3498db; color: white; border-radius: 8px; padding: 10px 20px;}
        .stButton>button:hover {background-color: #2980b9;}
        .tab-content {padding: 20px;}
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='subheader'>AI Medical Report & MCQ Analyzer</div>", unsafe_allow_html=True)
        st.write("Medical reports ya MCQ answer sheets ke liye input method chuno.")
        input_method = st.radio("Input Method:", ("Paste Medical Report Text", "Upload Medical Report File", "Upload MCQ Answer Sheet"))
        
        report_text = ""
        file_type = None
        mcq_answers = None
        
        if input_method == "Paste Medical Report Text":
            report_text = st.text_area("Apna medical report yahan paste karo:", height=200)
            file_type = "text"
        elif input_method == "Upload Medical Report File":
            uploaded_file = st.file_uploader("Upload karo (.txt, .pdf, .jpg, ya .png)", type=["txt", "pdf", "jpg", "png"])
            if uploaded_file:
                st.write(f"Uploaded file: {uploaded_file.name}")
                if uploaded_file.type == "application/pdf":
                    report_text = extract_text(uploaded_file, "pdf")
                    file_type = "pdf"
                elif uploaded_file.type in ["image/jpeg", "image/png"]:
                    report_text = extract_text(uploaded_file, "image")
                    file_type = "image"
                else:
                    report_text = uploaded_file.read().decode("utf-8")
                    file_type = "text"
        else:  # Upload MCQ Answer Sheet
            uploaded_file = st.file_uploader("MCQ answer sheet upload karo (.jpg ya .png)", type=["jpg", "png"])
            if uploaded_file:
                st.write(f"Uploaded file: {uploaded_file.name}")
                mcq_answers = extract_mcq_answers(uploaded_file)
                file_type = "mcq"
        
        if st.button("Analyze"):
            if input_method in ["Paste Medical Report Text", "Upload Medical Report File"]:
                if report_text and not report_text.startswith("Error"):
                    st.session_state["report_text"] = report_text
                    st.session_state["file_type"] = file_type
                else:
                    st.error(f"{file_type or 'file'} se text nahi nikla. Doosra file try karo ya text paste karo.")
                    st.info("Tips: File valid PDF ya high-resolution image honi chahiye. PDF text-based ho ya scanned. Image ke liye 300 DPI+ resolution use karo.")
            else:  # MCQ Answer Sheet
                if mcq_answers and not isinstance(mcq_answers, str):
                    st.session_state["mcq_answers"] = mcq_answers
                    st.session_state["file_type"] = file_type
                else:
                    st.error(mcq_answers if isinstance(mcq_answers, str) else "MCQ answers nahi nikle. Image mein clear question numbers aur answers hone chahiye.")
                    st.info("Tips: High-resolution image (300 DPI+) use karo jismein text clear ho, jaise '1. B'. Blurry ya low-contrast images avoid karo.")
    
    # Main Content with Tabs
    st.markdown("<div class='title'>AI Medical Report & MCQ Analyzer Assistant</div>", unsafe_allow_html=True)
    
    if "report_text" in st.session_state or "mcq_answers" in st.session_state:
        if st.session_state.get("file_type") == "mcq":
            tabs = st.tabs(["MCQ Results"])
            with tabs[0]:
                st.markdown("<div class='subheader'>MCQ Answers</div>", unsafe_allow_html=True)
                answers = st.session_state["mcq_answers"]
                st.write("**MCQ Answers Mile:**")
                for q, ans in sorted(answers.items(), key=lambda x: int(x[0][1:])):
                    st.write(f"{q}: {ans}")
        else:
            tabs = st.tabs(["Raw Report", "OCR Results", "Analysis Results", "Visualizations"])
            
            # Tab 1: Raw Report
            with tabs[0]:
                st.markdown("<div class='subheader'>Raw Input</div>", unsafe_allow_html=True)
                if st.session_state.get("file_type") in ["pdf", "image"]:
                    st.write("File se nikala gaya text:")
                st.markdown(f"<div class='raw-text'>{st.session_state['report_text']}</div>", unsafe_allow_html=True)
            
            # Tab 2: OCR Results
            with tabs[1]:
                st.markdown("<div class='subheader'>OCR Nikala Gaya Text</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='raw-text'>{st.session_state['report_text']}</div>", unsafe_allow_html=True)
            
            # Tab 3: Analysis Results
            with tabs[2]:
                with st.spinner("Report analyze ho raha hai..."):
                    summary, extracted_text = analyze_medical_report(st.session_state["report_text"])
                    
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown("<div class='subheader'>Analysis Results</div>", unsafe_allow_html=True)
                    
                    st.write("**Risk Summary:**")
                    st.write(summary["Risk Summary"])
                    
                    if summary["Lab Data"]:
                        st.write("**Lab Results:**")
                        df = pd.DataFrame(summary["Lab Data"])
                        st.dataframe(df.style.apply(lambda x: ['background: #ff3333' if x['Status'] == 'Critical' else '#ffcc00' if x['Status'] == 'Borderline' else '' for i in x], axis=1))
                    
                    st.write("**Diagnoses Identified:**")
                    for diag in summary["Diagnoses"]:
                        st.write(f"- {diag}")
                    
                    st.write("**Medications:**")
                    for med in summary["Medications"]:
                        st.write(f"- {med}")
                    
                    st.write("**Abnormal Findings:**")
                    for finding in summary["Abnormal Findings"]:
                        st.markdown(f"<span style='color: red;'>- {finding}</span>", unsafe_allow_html=True)
                    
                    st.write("**Explanations:**")
                    for exp in summary["Explanations"]:
                        with st.expander(f"{exp['Test']}"):
                            st.write(exp["Explanation"])
                    
                    st.write("**Follow-up Suggestions:**")
                    for suggestion in summary["Suggestions"]:
                        st.write(f"- {suggestion}")
                    
                    if summary["Lab Data"]:
                        st.write("**Export Results:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = pd.DataFrame(summary["Lab Data"]).to_csv(index=False)
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name="lab_results.csv",
                                mime="text/csv"
                            )
                        with col2:
                            pdf_buffer = generate_pdf_report(summary)
                            st.download_button(
                                label="Download as PDF",
                                data=pdf_buffer,
                                file_name="lab_results.pdf",
                                mime="application/pdf"
                            )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Tab 4: Visualizations
            with tabs[3]:
                st.markdown("<div class='subheader'>Lab Results Visualization</div>", unsafe_allow_html=True)
                if summary["Lab Data"]:
                    chart = create_bar_chart(summary["Lab Data"])
                    st.markdown(f"<img src='{chart}' style='width:100%;'>", unsafe_allow_html=True)
                else:
                    st.write("Visualization ke liye koi lab data nahi hai.")

if __name__ == "__main__":
    main()