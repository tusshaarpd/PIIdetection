#!/usr/bin/env python3
"""
PII Detection and Redaction Tool for PDF Documents
Detects and redacts various types of Personally Identifiable Information (PII)
"""

import re
import fitz  # PyMuPDF
import spacy
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class PIIDetector:
    """Main class for detecting various types of PII in text"""
    
    def __init__(self):
        """Initialize the PII detector with patterns and NLP model"""
        self.nlp = None
        self._load_nlp_model()
        self._compile_patterns()
        
    def _load_nlp_model(self):
        """Load spaCy NLP model for named entity recognition"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            logger.warning("Name detection will be limited without NLP model")
    
    def _compile_patterns(self):
        """Compile regex patterns for different types of PII"""
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'date_of_birth': re.compile(r'\b(?:0[1-9]|1[0-2])[/\-.](?:0[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b'),
            'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'driver_license': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),
            'passport': re.compile(r'\b[A-Z]{2}\d{7}\b'),
            'bank_account': re.compile(r'\b\d{8,17}\b'),
            'medical_record': re.compile(r'\bMRN:?\s*\d{6,10}\b', re.IGNORECASE),
            'patient_id': re.compile(r'\b(?:Patient|PT)\s*(?:ID|#):?\s*\d{6,10}\b', re.IGNORECASE),
        }
        
        # Address pattern (simplified)
        self.address_pattern = re.compile(
            r'\b\d+\s+(?:[A-Za-z]+\s+)*(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
            re.IGNORECASE
        )
    
    def detect_emails(self, text: str) -> List[PIIMatch]:
        """Detect email addresses"""
        matches = []
        for match in self.patterns['email'].finditer(text):
            matches.append(PIIMatch(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                pii_type='email',
                confidence=0.95,
                page_num=0
            ))
        return matches
    
    def detect_phone_numbers(self, text: str) -> List[PIIMatch]:
        """Detect phone numbers"""
        matches = []
        for match in self.patterns['phone'].finditer(text):
            matches.append(PIIMatch(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                pii_type='phone',
                confidence=0.9,
                page_num=0
            ))
        return matches
    
    def detect_ssn(self, text: str) -> List[PIIMatch]:
        """Detect Social Security Numbers"""
        matches = []
        for match in self.patterns['ssn'].finditer(text):
            # Validate SSN format
            ssn = re.sub(r'[^\d]', '', match.group())
            if len(ssn) == 9 and ssn != '000000000':
                matches.append(PIIMatch(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    pii_type='ssn',
                    confidence=0.98,
                    page_num=0
                ))
        return matches
    
    def detect_credit_cards(self, text: str) -> List[PIIMatch]:
        """Detect credit card numbers"""
        matches = []
        for match in self.patterns['credit_card'].finditer(text):
            # Basic Luhn algorithm check
            if self._is_valid_credit_card(match.group()):
                matches.append(PIIMatch(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    pii_type='credit_card',
                    confidence=0.95,
                    page_num=0
                ))
        return matches
    
    def detect_dates_of_birth(self, text: str) -> List[PIIMatch]:
        """Detect dates of birth"""
        matches = []
        for match in self.patterns['date_of_birth'].finditer(text):
            matches.append(PIIMatch(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                pii_type='date_of_birth',
                confidence=0.8,
                page_num=0
            ))
        return matches
    
    def detect_addresses(self, text: str) -> List[PIIMatch]:
        """Detect street addresses"""
        matches = []
        for match in self.address_pattern.finditer(text):
            matches.append(PIIMatch(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                pii_type='address',
                confidence=0.7,
                page_num=0
            ))
        return matches
    
    def detect_names(self, text: str) -> List[PIIMatch]:
        """Detect person names using NLP"""
        matches = []
        if not self.nlp:
            return matches
            
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                matches.append(PIIMatch(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    pii_type='name',
                    confidence=0.85,
                    page_num=0
                ))
        return matches
    
    def detect_other_pii(self, text: str) -> List[PIIMatch]:
        """Detect other types of PII"""
        matches = []
        
        # Detect various ID patterns
        for pii_type, pattern in self.patterns.items():
            if pii_type not in ['email', 'phone', 'ssn', 'credit_card', 'date_of_birth']:
                for match in pattern.finditer(text):
                    matches.append(PIIMatch(
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        pii_type=pii_type,
                        confidence=0.8,
                        page_num=0
                    ))
        
        return matches
    
    def detect_all_pii(self, text: str, page_num: int = 0) -> List[PIIMatch]:
        """Detect all types of PII in the given text"""
        all_matches = []
        
        # Detect different types of PII
        detection_methods = [
            self.detect_emails,
            self.detect_phone_numbers,
            self.detect_ssn,
            self.detect_credit_cards,
            self.detect_dates_of_birth,
            self.detect_addresses,
            self.detect_names,
            self.detect_other_pii
        ]
        
        for method in detection_methods:
            matches = method(text)
            for match in matches:
                match.page_num = page_num
                all_matches.append(match)
        
        # Remove overlapping matches (keep highest confidence)
        all_matches = self._remove_overlaps(all_matches)
        
        return all_matches
    
    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping PII matches, keeping the one with highest confidence"""
        if not matches:
            return matches
        
        # Sort by start position
        matches.sort(key=lambda x: x.start)
        
        result = []
        for current in matches:
            if not result:
                result.append(current)
                continue
            
            last = result[-1]
            
            # Check for overlap
            if current.start < last.end:
                # If overlap, keep the one with higher confidence
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
        
        # Luhn algorithm
        digits = [int(d) for d in number]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        return sum(digits) % 10 == 0

class PDFRedactor:
    """Class to handle PDF processing and redaction"""
    
    def __init__(self):
        self.detector = PIIDetector()
    
    def process_pdf(self, input_path: str, output_path: str = None, redaction_color: Tuple[float, float, float] = (0, 0, 0)) -> Dict:
        """
        Process PDF document to detect and redact PII
        
        Args:
            input_path: Path to input PDF file
            output_path: Path for output PDF file (optional)
            redaction_color: RGB color for redaction (default: black)
            
        Returns:
            Dictionary with processing results
        """
        if output_path is None:
            output_path = self._generate_output_path(input_path)
        
        results = {
            'input_file': input_path,
            'output_file': output_path,
            'total_pii_found': 0,
            'pii_by_type': {},
            'pii_by_page': {},
            'processing_time': 0,
            'status': 'success'
        }
        
        start_time = datetime.now()
        
        try:
            # Open PDF document
            doc = fitz.open(input_path)
            logger.info(f"Processing PDF with {len(doc)} pages")
            
            all_pii_matches = []
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Detect PII on this page
                pii_matches = self.detector.detect_all_pii(text, page_num)
                
                if pii_matches:
                    logger.info(f"Found {len(pii_matches)} PII items on page {page_num + 1}")
                    
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
            
            # Save redacted PDF
            doc.save(output_path)
            doc.close()
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")
            logger.info(f"Redacted PDF saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _generate_output_path(self, input_path: str) -> str:
        """Generate output file path"""
        path = Path(input_path)
        return str(path.parent / f"{path.stem}_redacted{path.suffix}")
    
    def _find_text_rect(self, page, text: str) -> Tuple[float, float, float, float]:
        """Find rectangle coordinates of text on page"""
        text_instances = page.search_for(text)
        if text_instances:
            return text_instances[0]  # Return first occurrence
        return None
    
    def _redact_page(self, page, pii_matches: List[PIIMatch], color: Tuple[float, float, float]):
        """Redact PII items on a page"""
        for match in pii_matches:
            if match.rect:
                # Create redaction annotation
                redact_area = fitz.Rect(match.rect)
                page.add_redact_annot(redact_area)
        
        # Apply redactions
        page.apply_redactions()
    
    def _group_by_type(self, matches: List[PIIMatch]) -> Dict:
        """Group PII matches by type"""
        grouped = {}
        for match in matches:
            if match.pii_type not in grouped:
                grouped[match.pii_type] = []
            grouped[match.pii_type].append({
                'text': match.text,
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
                'text': match.text,
                'type': match.pii_type,
                'confidence': match.confidence
            })
        return grouped

def main():
    """Main function to run the PII redaction tool"""
    parser = argparse.ArgumentParser(description='PII Detection and Redaction Tool for PDF Documents')
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument('-o', '--output', help='Path to output PDF file (optional)')
    parser.add_argument('-r', '--report', help='Path to save detection report as JSON')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not Path(args.input_pdf).exists():
        logger.error(f"Input file not found: {args.input_pdf}")
        return
    
    # Initialize redactor and process PDF
    redactor = PDFRedactor()
    results = redactor.process_pdf(args.input_pdf, args.output)
    
    # Print summary
    print(f"\n{'='*50}")
    print("PII REDACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Input file: {results['input_file']}")
    print(f"Output file: {results['output_file']}")
    print(f"Total PII found: {results['total_pii_found']}")
    print(f"Processing time: {results['processing_time']:.2f} seconds")
    print(f"Status: {results['status']}")
    
    if results['pii_by_type']:
        print(f"\nPII by type:")
        for pii_type, items in results['pii_by_type'].items():
            print(f"  {pii_type}: {len(items)} items")
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.report}")

if __name__ == "__main__":
    main()
