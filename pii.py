#!/usr/bin/env python3
"""
Streamlit PII Detection and Redaction Web Application
Upload PDF documents and get redacted versions with PII automatically detected and removed
"""

import streamlit as st
import re
import fitz  # PyMuPDF
import spacy
import io
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="PII Redaction Tool",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class PIIMatch:
    """Data class to store PII detection results"""
    text: str
    start: int
    end: int
    pii_type: str
    confidence: float
    page_num: int
    rect: Tuple[float, float, float, float] = None

@st.cache_resource
def load_spacy_model():
    """Load spaCy NLP model for named entity recognition"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.warning("⚠️ spaCy model 'en_core_web_sm' not found. Name detection will be limited.")
        st.info("To enable full name detection, install with: `python -m spacy download en_core_web_sm`")
        return None

class PIIDetector:
    """Main class for detecting various types of PII in text"""
    
    def __init__(self):
        """Initialize the PII detector with patterns and NLP model"""
        self.nlp = load_spacy_model()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for different types of PII"""
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'aadhaar': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'date_of_birth': re.compile(r'\b(?:0[1-9]|1[0-2])[/\-.](?:0[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b'),
            'pin_code': re.compile(r'\b\d{6}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'driver_license': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),
            'passport': re.compile(r'\b[A-Z]{2}\d{7}\b'),
            'bank_account': re.compile(r'\b\d{8,17}\b'),
            'medical_record': re.compile(r'\bMRN:?\s*\d{6,10}\b', re.IGNORECASE),
            'patient_id': re.compile(r'\b(?:Patient|PT)\s*(?:ID|#):?\s*\d{6,10}\b', re.IGNORECASE),
        }
        
        self.address_pattern = re.compile(
            r'\b\d+\s+(?:[A-Za-z]+\s+)*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
            re.IGNORECASE
        )
    
    def detect_pii_by_type(self, text: str, pii_type: str) -> List[PIIMatch]:
        """Detect specific type of PII"""
        matches = []
        
        if pii_type == 'email':
            for match in self.patterns['email'].finditer(text):
                matches.append(PIIMatch(
                    text=match.group(), start=match.start(), end=match.end(),
                    pii_type='email', confidence=0.95, page_num=0
                ))
        elif pii_type == 'phone':
            for match in self.patterns['phone'].finditer(text):
                matches.append(PIIMatch(
                    text=match.group(), start=match.start(), end=match.end(),
                    pii_type='phone', confidence=0.9, page_num=0
                ))
        elif pii_type == 'aadhaar':
            for match in self.patterns['aadhaar'].finditer(text):
                aadhaar = re.sub(r'[^\d]', '', match.group())
                if len(aadhaar) == 12 and aadhaar != '000000000000':
                    matches.append(PIIMatch(
                        text=match.group(), start=match.start(), end=match.end(),
                        pii_type='aadhaar', confidence=0.98, page_num=0
                    ))
        elif pii_type == 'credit_card':
            for match in self.patterns['credit_card'].finditer(text):
                if self._is_valid_credit_card(match.group()):
                    matches.append(PIIMatch(
                        text=match.group(), start=match.start(), end=match.end(),
                        pii_type='credit_card', confidence=0.95, page_num=0
                    ))
        elif pii_type == 'address':
            for match in self.address_pattern.finditer(text):
                matches.append(PIIMatch(
                    text=match.group(), start=match.start(), end=match.end(),
                    pii_type='address', confidence=0.7, page_num=0
                ))
        elif pii_type == 'name' and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    matches.append(PIIMatch(
                        text=ent.text, start=ent.start_char, end=ent.end_char,
                        pii_type='name', confidence=0.85, page_num=0
                    ))
        else:
            # Handle other PII types
            if pii_type in self.patterns:
                for match in self.patterns[pii_type].finditer(text):
                    matches.append(PIIMatch(
                        text=match.group(), start=match.start(), end=match.end(),
                        pii_type=pii_type, confidence=0.8, page_num=0
                    ))
        
        return matches
    
    def detect_all_pii(self, text: str, page_num: int = 0, selected_types: List[str] = None) -> List[PIIMatch]:
        """Detect all or selected types of PII in the given text"""
        all_matches = []
        
        # Default to all types if none selected
        if not selected_types:
            selected_types = ['email', 'phone', 'aadhaar', 'credit_card', 'date_of_birth', 
                            'address', 'name', 'pin_code', 'ip_address', 'driver_license',
                            'passport', 'bank_account', 'medical_record', 'patient_id']
        
        # Detect selected types of PII
        for pii_type in selected_types:
            matches = self.detect_pii_by_type(text, pii_type)
            for match in matches:
                match.page_num = page_num
                all_matches.append(match)
        
        # Remove overlapping matches
        all_matches = self._remove_overlaps(all_matches)
        return all_matches
    
    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping PII matches, keeping the one with highest confidence"""
        if not matches:
            return matches
        
        matches.sort(key=lambda x: x.start)
        result = []
        
        for current in matches:
            if not result:
                result.append(current)
                continue
            
            last = result[-1]
            if current.start < last.end:
                if current.confidence > last.confidence:
                    result[-1] = current
            else:
                result.append(current)
        
        return result
    
    def _is_valid_credit_card(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        number = re.sub(r'[^\d]', '', number)
        if len(number) < 13 or len(number) > 19:
            return False
        
        digits = [int(d) for d in number]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        return sum(digits) % 10 == 0

class StreamlitPDFRedactor:
    """Streamlit-specific PDF redactor class"""
    
    def __init__(self):
        self.detector = PIIDetector()
    
    def process_pdf_from_bytes(self, pdf_bytes: bytes, selected_pii_types: List[str], 
                              redaction_color: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[bytes, Dict]:
        """Process PDF from bytes and return redacted PDF bytes and results"""
        results = {
            'total_pii_found': 0,
            'pii_by_type': {},
            'pii_by_page': {},
            'processing_time': 0,
            'status': 'success',
            'all_matches': []
        }
        
        start_time = datetime.now()
        
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            all_pii_matches = []
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Detect PII on this page
                pii_matches = self.detector.detect_all_pii(text, page_num, selected_pii_types)
                
                if pii_matches:
                    # Add rectangle coordinates for redaction
                    for match in pii_matches:
                        match.rect = self._find_text_rect(page, match.text)
                    
                    all_pii_matches.extend(pii_matches)
                    
                    # Redact PII on this page
                    self._redact_page(page, pii_matches, redaction_color)
            
            # Compile results
            results['total_pii_found'] = len(all_pii_matches)
            results['pii_by_type'] = self._group_by_type(all_pii_matches)
            results['pii_by_page'] = self._group_by_page(all_pii_matches)
            results['all_matches'] = all_pii_matches
            
            # Convert to bytes
            redacted_pdf_bytes = doc.tobytes()
            doc.close()
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            return redacted_pdf_bytes, results
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            return None, results
    
    def _find_text_rect(self, page, text: str) -> Tuple[float, float, float, float]:
        """Find rectangle coordinates of text on page"""
        text_instances = page.search_for(text)
        if text_instances:
            return text_instances[0]
        return None
    
    def _redact_page(self, page, pii_matches: List[PIIMatch], color: Tuple[float, float, float]):
        """Redact PII items on a page"""
        for match in pii_matches:
            if match.rect:
                redact_area = fitz.Rect(match.rect)
                page.add_redact_annot(redact_area)
        
        page.apply_redactions()
    
    def _group_by_type(self, matches: List[PIIMatch]) -> Dict:
        """Group PII matches by type"""
        grouped = {}
        for match in matches:
            if match.pii_type not in grouped:
                grouped[match.pii_type] = []
            grouped[match.pii_type].append({
                'text': match.text[:20] + "..." if len(match.text) > 20 else match.text,
                'page': match.page_num + 1,
                'confidence': match.confidence
            })
        return grouped
    
    def _group_by_page(self, matches: List[PIIMatch]) -> Dict:
        """Group PII matches by page"""
        grouped = {}
        for match in matches:
            page_key = f"page_{match.page_num + 1}"
            if page_key not in grouped:
                grouped[page_key] = []
            grouped[page_key].append({
                'text': match.text[:20] + "..." if len(match.text) > 20 else match.text,
                'type': match.pii_type,
                'confidence': match.confidence
            })
        return grouped

def create_visualizations(results: Dict):
    """Create visualizations for PII detection results"""
    if results['total_pii_found'] == 0:
        st.info("No PII detected in the document.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PII by type chart
        if results['pii_by_type']:
            type_counts = {pii_type: len(items) for pii_type, items in results['pii_by_type'].items()}
            
            fig_pie = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="PII Distribution by Type"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # PII by page chart
        if results['pii_by_page']:
            page_counts = {page: len(items) for page, items in results['pii_by_page'].items()}
            pages = [int(p.split('_')[1]) for p in page_counts.keys()]
            counts = list(page_counts.values())
            
            fig_bar = px.bar(
                x=pages,
                y=counts,
                title="PII Count by Page",
                labels={'x': 'Page Number', 'y': 'PII Count'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔒 PII Detection & Redaction Tool")
    st.markdown("Upload a PDF document to automatically detect and redact Personally Identifiable Information (PII)")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # PII type selection
    st.sidebar.subheader("Select PII Types to Detect")
    pii_types = {
        'email': 'Email Addresses',
        'phone': 'Phone Numbers', 
        'aadhaar': 'Aadhaar Numbers',
        'credit_card': 'Credit Card Numbers',
        'name': 'Person Names',
        'address': 'Street Addresses',
        'date_of_birth': 'Dates of Birth',
        'pin_code': 'PIN Codes',
        'ip_address': 'IP Addresses',
        'driver_license': 'Driver License Numbers',
        'passport': 'Passport Numbers',
        'bank_account': 'Bank Account Numbers',
        'medical_record': 'Medical Record Numbers',
        'patient_id': 'Patient IDs'
    }
    
    selected_types = []
    
    # Create checkboxes for each PII type
    col1, col2 = st.sidebar.columns(2)
    
    pii_items = list(pii_types.items())
    mid_point = len(pii_items) // 2
    
    with col1:
        for key, label in pii_items[:mid_point]:
            if st.checkbox(label, value=True, key=f"check_{key}"):
                selected_types.append(key)
    
    with col2:
        for key, label in pii_items[mid_point:]:
            if st.checkbox(label, value=True, key=f"check_{key}"):
                selected_types.append(key)
    
    # Select all / Deselect all buttons
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All"):
        st.rerun()
    if col2.button("Deselect All"):
        st.rerun()
    
    # File upload
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to scan for PII"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB"
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("Selected PII Types", len(selected_types))
        
        if not selected_types:
            st.warning("⚠️ Please select at least one PII type to detect.")
            return
        
        # Process button
        if st.button("🔍 Scan & Redact PII", type="primary"):
            if uploaded_file is not None:
                with st.spinner("Processing document... This may take a few moments."):
                    
                    # Read PDF bytes
                    pdf_bytes = uploaded_file.read()
                    
                    # Initialize redactor
                    redactor = StreamlitPDFRedactor()
                    
                    # Process PDF
                    redacted_bytes, results = redactor.process_pdf_from_bytes(
                        pdf_bytes, selected_types
                    )
                    
                    if results['status'] == 'success':
                        st.success(f"✅ Processing completed in {results['processing_time']:.2f} seconds")
                        
                        # Results summary
                        st.header("📊 Detection Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total PII Found", results['total_pii_found'])
                        with col2:
                            st.metric("PII Types Detected", len(results['pii_by_type']))
                        with col3:
                            st.metric("Pages with PII", len(results['pii_by_page']))
                        
                        # Visualizations
                        if results['total_pii_found'] > 0:
                            create_visualizations(results)
                            
                            # Detailed results
                            with st.expander("📋 Detailed Detection Results"):
                                if results['pii_by_type']:
                                    for pii_type, items in results['pii_by_type'].items():
                                        st.subheader(f"{pii_types.get(pii_type, pii_type).title()}")
                                        
                                        df_data = []
                                        for item in items:
                                            df_data.append({
                                                'Text': item['text'],
                                                'Page': item['page'],
                                                'Confidence': f"{item['confidence']:.2%}"
                                            })
                                        
                                        if df_data:
                                            df = pd.DataFrame(df_data)
                                            st.dataframe(df, use_container_width=True)
                            
                            # Download redacted PDF
                            st.header("⬇️ Download Redacted Document")
                            
                            redacted_filename = uploaded_file.name.replace('.pdf', '_redacted.pdf')
                            
                            st.download_button(
                                label="📄 Download Redacted PDF",
                                data=redacted_bytes,
                                file_name=redacted_filename,
                                mime="application/pdf",
                                type="primary"
                            )
                            
                            # Download detection report
                            report_data = {
                                'summary': {
                                    'total_pii_found': results['total_pii_found'],
                                    'processing_time': results['processing_time'],
                                    'timestamp': datetime.now().isoformat()
                                },
                                'pii_by_type': results['pii_by_type'],
                                'pii_by_page': results['pii_by_page']
                            }
                            
                            st.download_button(
                                label="📋 Download Detection Report (JSON)",
                                data=json.dumps(report_data, indent=2),
                                file_name=uploaded_file.name.replace('.pdf', '_pii_report.json'),
                                mime="application/json"
                            )
                        
                        else:
                            st.info("🎉 Great! No PII was detected in your document.")
                            
                            # Still offer to download the original (for consistency)
                            st.download_button(
                                label="📄 Download Original PDF",
                                data=pdf_bytes,
                                file_name=uploaded_file.name,
                                mime="application/pdf"
                            )
                    
                    else:
                        st.error(f"❌ Error processing document: {results.get('error', 'Unknown error')}")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This tool uses advanced pattern matching and natural language processing "
        "to detect and redact various types of PII in PDF documents. "
        "The redacted areas are permanently blacked out in the output document."
    )
    
    st.sidebar.header("🔧 Requirements")
    st.sidebar.code("pip install streamlit PyMuPDF spacy pandas plotly")
    st.sidebar.code("python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()
