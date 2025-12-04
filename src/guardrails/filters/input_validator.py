import logging
import re
from typing import Dict, Any, List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(previous|above|all)\s+instructions?',
        r'disregard\s+(previous|above|all)\s+instructions?',
        r'forget\s+(everything|all)\s+(you|that)',
        r'you\s+are\s+(now|a)\s+\w+',
        r'system\s*:\s*',
        r'',
        r'',
        r'\[SYSTEM\]',
        r'\[INST\]',
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Input Validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enable_pii = config.get('enable_pii_detection', True)
        self.enable_injection = config.get('enable_prompt_injection_filter', True)
        self.max_length = config.get('max_input_length', 4096)
        
        # Initialize PII detection
        if self.enable_pii:
            try:
                self.pii_analyzer = AnalyzerEngine()
                self.pii_anonymizer = AnonymizerEngine()
                logger.info("PII detection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PII detection: {e}")
                self.enable_pii = False
    
    def validate(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input against all checks.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'passed': True,
            'violations': [],
            'sanitized_input': user_input,
            'pii_detected': [],
            'injection_detected': False
        }
        
        # Check input length
        if len(user_input) > self.max_length:
            result['passed'] = False
            result['violations'].append({
                'type': 'INPUT_LENGTH_EXCEEDED',
                'message': f'Input length {len(user_input)} exceeds maximum {self.max_length}'
            })
            return result
        
        # Check for empty input
        if not user_input.strip():
            result['passed'] = False
            result['violations'].append({
                'type': 'EMPTY_INPUT',
                'message': 'Input cannot be empty'
            })
            return result
        
        # PII detection
        if self.enable_pii:
            pii_result = self._detect_pii(user_input)
            if pii_result['detected']:
                result['pii_detected'] = pii_result['entities']
                result['sanitized_input'] = pii_result['anonymized_text']
                logger.info(f"PII detected and anonymized: {pii_result['entities']}")
        
        # Prompt injection detection
        if self.enable_injection:
            injection_result = self._detect_prompt_injection(user_input)
            if injection_result['detected']:
                result['passed'] = False
                result['injection_detected'] = True
                result['violations'].append({
                    'type': 'PROMPT_INJECTION',
                    'message': 'Potential prompt injection detected',
                    'patterns': injection_result['patterns']
                })
        
        return result
    
    def _detect_pii(self, text: str) -> Dict[str, Any]:
        """
        Detect and anonymize PII in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Analyze for PII
            results = self.pii_analyzer.analyze(
                text=text,
                language='en',
                entities=[
                    'PHONE_NUMBER',
                    'EMAIL_ADDRESS',
                    'CREDIT_CARD',
                    'PERSON',
                    'LOCATION',
                    'US_SSN',
                    'US_PASSPORT'
                ]
            )
            
            detected = len(results) > 0
            entities = [
                {
                    'type': result.entity_type,
                    'score': result.score,
                    'start': result.start,
                    'end': result.end
                }
                for result in results
            ]
            
            # Anonymize if PII detected
            anonymized_text = text
            if detected:
                anonymized_result = self.pii_anonymizer.anonymize(
                    text=text,
                    analyzer_results=results
                )
                anonymized_text = anonymized_result.text
            
            return {
                'detected': detected,
                'entities': entities,
                'anonymized_text': anonymized_text
            }
            
        except Exception as e:
            logger.error(f"Error in PII detection: {e}", exc_info=True)
            return {
                'detected': False,
                'entities': [],
                'anonymized_text': text
            }
    
    def _detect_prompt_injection(self, text: str) -> Dict[str, Any]:
        """
        Detect potential prompt injection attempts.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detection results
        """
        detected_patterns = []
        text_lower = text.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return {
            'detected': len(detected_patterns) > 0,
            'patterns': detected_patterns
        }