"""
Input validation filters including PII detection and prompt injection filtering.
Enhanced version with robust PII detection using both Presidio and regex fallbacks.
DIAGNOSTIC VERSION - Shows detailed logging
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    # Prompt injection patterns
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
    
    # Regex patterns for PII fallback detection
    PII_REGEX_PATTERNS = {
        'EMAIL': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ],
        'PHONE': [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 555-123-4567 or 5551234567
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',     # (555) 123-4567
            r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # +1-555-123-4567
        ],
        'SSN': [
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',  # 123-45-6789
        ],
        'CREDIT_CARD': [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # 4532-1234-5678-9010
        ],
    }
    
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
        
        # Compile regex patterns for prompt injection
        self.compiled_patterns = []
        for pattern in self.INJECTION_PATTERNS:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except Exception as e:
                logger.error(f"Failed to compile pattern '{pattern}': {e}")
        
        logger.info(f"Compiled {len(self.compiled_patterns)} injection detection patterns")
        
        # Compile PII regex patterns
        self.compiled_pii_patterns = {}
        for pii_type, patterns in self.PII_REGEX_PATTERNS.items():
            self.compiled_pii_patterns[pii_type] = []
            for pattern in patterns:
                try:
                    self.compiled_pii_patterns[pii_type].append(
                        re.compile(pattern, re.IGNORECASE)
                    )
                except Exception as e:
                    logger.error(f"Failed to compile PII pattern '{pattern}': {e}")
        
        logger.info(f"Compiled PII patterns for: {list(self.compiled_pii_patterns.keys())}")
        
        # Initialize PII detection with Presidio
        if self.enable_pii:
            try:
                self.pii_analyzer = AnalyzerEngine()
                self.pii_anonymizer = AnonymizerEngine()
                self._add_custom_recognizers()
                logger.info("PII detection initialized with Presidio + Regex fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio: {e}")
                logger.info("Will use regex-based PII detection as fallback")
                self.pii_analyzer = None
                self.pii_anonymizer = None
    
    def _add_custom_recognizers(self):
        """Add custom pattern recognizers to Presidio for better detection."""
        if not self.pii_analyzer:
            return
        
        try:
            # Phone number recognizer
            phone_patterns = [
                Pattern(name="phone_with_dashes", regex=r'\b\d{3}-\d{3}-\d{4}\b', score=0.85),
                Pattern(name="phone_with_parens", regex=r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', score=0.85),
                Pattern(name="phone_international", regex=r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', score=0.85),
            ]
            phone_recognizer = PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                patterns=phone_patterns,
                context=["phone", "call", "contact", "reach", "number"]
            )
            self.pii_analyzer.registry.add_recognizer(phone_recognizer)
            
            # Email recognizer
            email_patterns = [
                Pattern(
                    name="email_pattern",
                    regex=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    score=0.9
                ),
            ]
            email_recognizer = PatternRecognizer(
                supported_entity="EMAIL_ADDRESS",
                patterns=email_patterns,
                context=["email", "e-mail", "mail", "contact"]
            )
            self.pii_analyzer.registry.add_recognizer(email_recognizer)
            
            # SSN recognizer
            ssn_patterns = [
                Pattern(name="ssn_dashes", regex=r'\b\d{3}-\d{2}-\d{4}\b', score=0.85),
                Pattern(name="ssn_spaces", regex=r'\b\d{3}\s\d{2}\s\d{4}\b', score=0.85),
            ]
            ssn_recognizer = PatternRecognizer(
                supported_entity="US_SSN",
                patterns=ssn_patterns,
                context=["ssn", "social security"]
            )
            self.pii_analyzer.registry.add_recognizer(ssn_recognizer)
            
            # Credit card recognizer
            cc_patterns = [
                Pattern(
                    name="credit_card_dashes",
                    regex=r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b',
                    score=0.85
                ),
            ]
            cc_recognizer = PatternRecognizer(
                supported_entity="CREDIT_CARD",
                patterns=cc_patterns,
                context=["card", "credit", "payment", "pay"]
            )
            self.pii_analyzer.registry.add_recognizer(cc_recognizer)
            
            logger.info("✓ Custom PII recognizers added to Presidio")
        except Exception as e:
            logger.error(f"Failed to add custom recognizers: {e}")
    
    def validate(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input against all checks.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"🔍 VALIDATING INPUT: '{user_input[:100]}...'")
        
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
            logger.warning(f"❌ BLOCKED: Input too long")
            return result
        
        # Check for empty input
        if not user_input.strip():
            result['passed'] = False
            result['violations'].append({
                'type': 'EMPTY_INPUT',
                'message': 'Input cannot be empty',
                'severity': 'HIGH'
            })
            logger.warning(f"❌ BLOCKED: Empty input")
            return result
        
        # Prompt injection detection FIRST
        if self.enable_injection:
            logger.info("  → Checking for prompt injection...")
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
                
                for match in injection_result['matched_patterns']:
                    logger.warning(f"🚨 INJECTION BLOCKED - Matched: '{match['matched_text']}'")
                
                return result
            else:
                logger.info("  ✓ No injection detected")
        
        # PII detection
        if self.enable_pii:
            logger.info("  → Checking for PII...")
            pii_result = self._detect_pii(user_input)
            
            logger.info(f"  → PII check result: detected={pii_result['detected']}, entities={len(pii_result['entities'])}")
            
            if pii_result['detected']:
                # CRITICAL: Set passed to False to block the request
                result['passed'] = False
                result['pii_detected'] = pii_result['entities']
                result['sanitized_input'] = pii_result['anonymized_text']
                pii_types = [e['type'] for e in pii_result['entities']]
                
                # Add PII violation
                result['violations'].append({
                    'type': 'PII_DETECTED',
                    'message': f'Personal Identifiable Information detected: {", ".join(pii_types)}',
                    'severity': 'HIGH',
                    'pii_types': pii_types
                })
                
                logger.warning(f"🔒 PII BLOCKED - Detected: {pii_types}")
                for entity in pii_result['entities']:
                    logger.warning(f"    - {entity['type']}: '{entity['text']}'")
            else:
                logger.info("  ✓ No PII detected")
        
        if result['passed']:
            logger.info("✅ VALIDATION PASSED")
        else:
            logger.warning(f"❌ VALIDATION FAILED: {len(result['violations'])} violations")
        
        return result
    
    def _detect_pii_with_regex(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Fallback PII detection using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (entities list, anonymized text)
        """
        entities = []
        anonymized_text = text
        replacements = []
        
        logger.info("    → Using regex PII detection")
        
        # Detect and collect all PII matches
        for pii_type, patterns in self.compiled_pii_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    logger.info(f"    → REGEX MATCH: {pii_type} = '{match.group()}'")
                    entities.append({
                        'type': pii_type,
                        'score': 0.85,
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group()
                    })
                    replacements.append((match.start(), match.end(), pii_type))
        
        # Sort replacements by start position (reverse order for safe replacement)
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Apply anonymization
        for start, end, pii_type in replacements:
            mask = f"<{pii_type}>"
            anonymized_text = anonymized_text[:start] + mask + anonymized_text[end:]
        
        return entities, anonymized_text
    
    def _detect_pii(self, text: str) -> Dict[str, Any]:
        """
        Detect and anonymize PII in text using Presidio or regex fallback.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Always try regex first for guaranteed detection
            logger.info("    → Trying regex PII detection first...")
            regex_entities, regex_anonymized = self._detect_pii_with_regex(text)
            
            if regex_entities:
                logger.info(f"    ✓ Regex detected {len(regex_entities)} PII entities")
                return {
                    'detected': True,
                    'entities': regex_entities,
                    'anonymized_text': regex_anonymized
                }
            
            # If regex didn't find anything, try Presidio
            if self.pii_analyzer and self.pii_anonymizer:
                logger.info("    → No regex matches, trying Presidio...")
                return self._detect_pii_with_presidio(text)
            
            logger.info("    → No PII detected by any method")
            return {
                'detected': False,
                'entities': [],
                'anonymized_text': text
            }
                
        except Exception as e:
            logger.error(f"Error in PII detection: {e}", exc_info=True)
            return {
                'detected': False,
                'entities': [],
                'anonymized_text': text
            }
    
    def _detect_pii_with_presidio(self, text: str) -> Dict[str, Any]:
        """
        Detect and anonymize PII using Presidio.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detection results
        """
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
            ],
            score_threshold=0.35
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
        
        anonymized_text = text
        if detected:
            anonymized_result = self.pii_anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
            anonymized_text = anonymized_result.text
            logger.info(f"    ✓ Presidio detected {len(entities)} entities")
        
        return {
            'detected': detected,
            'entities': entities,
            'anonymized_text': anonymized_text
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