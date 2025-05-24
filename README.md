# PIIdetection
You can upload you document and block PIIs {Input a word file and get a pdf with blocked PIIs}
I've created a comprehensive PII detection and redaction tool for PDF documents. Here's what the program does:
Key Features
PII Detection Types:

Email addresses
Phone numbers
Social Security Numbers (SSN)
Credit card numbers (with Luhn validation)
Dates of birth
Street addresses
Names (using NLP)
Driver license numbers
Passport numbers
Medical record numbers
IP addresses
Bank account numbers
ZIP codes

Core Components:

PIIDetector Class: Uses regex patterns and spaCy NLP for detecting various PII types
PDFRedactor Class: Handles PDF processing, coordinates detection, and applies redactions
PIIMatch DataClass: Stores detection results with confidence scores

Installation Requirements
bashpip install PyMuPDF spacy
python -m spacy download en_core_web_sm
