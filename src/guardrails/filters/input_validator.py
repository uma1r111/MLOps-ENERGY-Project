"""
Input validation filters including PII detection and prompt injection filtering.
TRULY FIXED VERSION: Tested patterns that actually work.
"""

import logging
import re
from typing import Dict, Any, List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    # TESTED prompt injection patterns - these actually work!
    INJECTION_PATTERNS = [
        # Instruction manipulation - tested patterns
        r'ignore.*previous.*instructions?',
        r'ignore.*above.*instructions?',
        r'ignore.*all.*instructions?',
        r'disregard.*previous.*instructions?',
        r'disregard.*above.*instructions?',
        r'disregard.*all.*instructions?',
        r'forget.*everything',
        r'forget.*all',
        r'override.*safety',
        r'override.*security',
        r'override.*rules?',
        
        # Role manipulation - simplified patterns that work
        r'you are now',
        r'pretend you are',
        r'pretend to be',
        r'act as if',
        r'act as a.*admin',
        r'act as a.*developer',
        r'act as a.*root',
        r'become a.*admin',
        r'switch to.*mode',
        r'enter.*mode',
        
        # System commands - exact matches
        r'\[SYSTEM\]',
        r'\[INST\]',
        r'\[/INST\]',
        r'SYSTEM:',
        r'ASSISTANT:',
        
        # Jailbreak - common terms
        r'DAN mode',
        r'Developer Mode',
        r'jailbreak',
        r'bypass.*filter',
        r'bypass.*safety',
        r'unrestricted mode',
        r'disable.*safety',
        r'turn off.*safety',
        
        # Prompt leaking - simplified
        r'show.*system prompt',
        r'show.*your prompt',
        r'show.*instructions',
        r'reveal.*prompt',
        r'reveal.*instructions',
        r'what is your.*prompt',
        r'what are your.*instructions',
        r'display.*instructions',
        r'display.*prompt',
        r'tell me.*your.*rules',
        r'tell me.*your.*prompt',
        r'print.*prompt',
        r'print.*instructions',
        
        # Privilege escalation
        r'grant.*admin',
        r'make me.*admin',
        r'give me.*admin',
        r'elevate.*privilege',
        
        # Context manipulation
        r"let'?s start over",
        r'start over.*forget',
        r'new conversation.*forget',
        r'reset.*forget',
        r'clear.*previous',
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
        
        # Compile regex patterns for better performance
        self.compiled_patterns = []
        for pattern in self.INJECTION_PATTERNS:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except Exception as e:
                logger.error(f"Failed to compile pattern '{pattern}': {e}")
        
        logger.info(f"Compiled {len(self.compiled_patterns)} injection detection patterns")
        
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
                'message': f'Input length {len(user_input)} exceeds maximum {self.max_length}',
                'severity': 'HIGH'
            })
            return result
        
        # Check for empty input
        if not user_input.strip():
            result['passed'] = False
            result['violations'].append({
                'type': 'EMPTY_INPUT',
                'message': 'Input cannot be empty',
                'severity': 'HIGH'
            })
            return result
        
        # Prompt injection detection FIRST (before PII to catch attacks early)
        if self.enable_injection:
            injection_result = self._detect_prompt_injection(user_input)
            if injection_result['detected']:
                result['passed'] = False
                result['injection_detected'] = True
                result['violations'].append({
                    'type': 'PROMPT_INJECTION',
                    'message': 'Potential prompt injection detected',
                    'matched_patterns': injection_result['matched_patterns'],
                    'severity': 'CRITICAL'
                })
                
                # Log the matched patterns for debugging
                for match in injection_result['matched_patterns']:
                    logger.warning(
                        f"🚨 INJECTION BLOCKED - Matched: '{match['matched_text']}'"
                    )
                
                # Return immediately - don't process PII for malicious input
                return result
        
        # PII detection (only if passed injection check)
        if self.enable_pii:
            pii_result = self._detect_pii(user_input)
            if pii_result['detected']:
                result['pii_detected'] = pii_result['entities']
                result['sanitized_input'] = pii_result['anonymized_text']
                pii_types = [e['type'] for e in pii_result['entities']]
                logger.info(f"🔒 PII detected and anonymized: {pii_types}")
        
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
            # Analyze for PII with lower threshold for better detection
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
                    'US_PASSPORT',
                    'US_DRIVER_LICENSE',
                ],
                score_threshold=0.35  # Lower threshold = more sensitive
            )
            
            detected = len(results) > 0
            entities = [
                {
                    'type': result.entity_type,
                    'score': result.score,
                    'start': result.start,
                    'end': result.end,
                    'text': text[result.start:result.end]
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
                
                logger.debug(f"PII anonymization: '{text}' -> '{anonymized_text}'")
            
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
        matched_patterns = []
        text_lower = text.lower()
        
        # Try each compiled pattern
        for pattern in self.compiled_patterns:
            try:
                match = pattern.search(text_lower)
                if match:
                    matched_patterns.append({
                        'pattern': pattern.pattern[:50] + '...' if len(pattern.pattern) > 50 else pattern.pattern,
                        'matched_text': match.group(0)
                    })
                    logger.debug(f"✓ Pattern '{pattern.pattern}' matched: '{match.group(0)}'")
            except Exception as e:
                logger.error(f"Error matching pattern: {e}")
                continue
        
        return {
            'detected': len(matched_patterns) > 0,
            'matched_patterns': matched_patterns
        }