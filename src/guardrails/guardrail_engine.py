"""
Main Guardrail Engine for orchestrating all safety checks.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path

from .filters.input_validator import InputValidator
from .filters.output_moderator import OutputModerator
from .policies.policy_engine import PolicyEngine
from .metrics import GuardrailMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GuardrailEngine:
    """
    Orchestrates all guardrail checks for LLM inputs and outputs.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        log_dir: str = "logs/guardrails"
    ):
        """
        Initialize the Guardrail Engine.
        
        Args:
            config_path: Path to guardrail configuration file
            log_dir: Directory for guardrail event logs
        """
        self.config = self._load_config(config_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.input_validator = InputValidator(self.config.get('input', {}))
        self.output_moderator = OutputModerator(self.config.get('output', {}))
        self.policy_engine = PolicyEngine(self.config.get('policies', {}))
        
        logger.info("GuardrailEngine initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load guardrail configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'input': {
                'enable_pii_detection': True,
                'enable_prompt_injection_filter': True,
                'enable_content_filter': True,
                'max_input_length': 4096
            },
            'output': {
                'enable_toxicity_filter': True,
                'enable_hallucination_filter': True,
                'toxicity_threshold': 0.7,
                'max_output_length': 2048
            },
            'policies': {
                'block_on_violation': True,
                'log_all_events': True
            }
        }
    
    def validate_input(
        self,
        user_input: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate user input against all input guardrails.
        
        Args:
            user_input: The user's input text
            metadata: Additional metadata for the request
            
        Returns:
            Dictionary with validation results and any violations
        """
        start_time = time.time()
        
        # Initialize with all expected fields
        validation_result = {
            'passed': True,
            'violations': [],
            'sanitized_input': user_input,
            'pii_detected': [],  # Always include
            'injection_detected': False,  # Always include
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat(),
            'latency_ms': 0
        }
        
        try:
            # Run input validation checks
            input_result = self.input_validator.validate(user_input)
            
            # Merge all fields from input_result
            validation_result['pii_detected'] = input_result.get('pii_detected', [])
            validation_result['injection_detected'] = input_result.get('injection_detected', False)
            validation_result['sanitized_input'] = input_result.get('sanitized_input', user_input)
            
            if not input_result['passed']:
                validation_result['passed'] = False
                validation_result['violations'].extend(input_result['violations'])
            
            # Check against policies
            policy_result = self.policy_engine.check_input_policy(
                user_input, input_result
            )
            
            if not policy_result['passed']:
                validation_result['passed'] = False
                validation_result['violations'].extend(policy_result['violations'])
            
        except Exception as e:
            logger.error(f"Error during input validation: {e}", exc_info=True)
            validation_result['passed'] = False
            validation_result['violations'].append({
                'type': 'SYSTEM_ERROR',
                'message': str(e)
            })
        
        validation_result['latency_ms'] = (time.time() - start_time) * 1000
        
        # Log the event
        self._log_event('input_validation', validation_result)
        
        return validation_result
    
    def moderate_output(
        self,
        model_output: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Moderate model output against all output guardrails.
        
        Args:
            model_output: The model's generated output
            metadata: Additional metadata for the request
            
        Returns:
            Dictionary with moderation results and any violations
        """
        start_time = time.time()
        
        # Initialize with all expected fields
        moderation_result = {
            'passed': True,
            'violations': [],
            'sanitized_output': model_output,
            'toxicity_scores': {},  # Always include
            'hallucination_detected': False,  # Always include
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat(),
            'latency_ms': 0
        }
        
        try:
            # Run output moderation checks
            output_result = self.output_moderator.moderate(model_output)
            
            # Merge all fields from output_result
            moderation_result['toxicity_scores'] = output_result.get('toxicity_scores', {})
            moderation_result['hallucination_detected'] = output_result.get('hallucination_detected', False)
            moderation_result['sanitized_output'] = output_result.get('sanitized_output', model_output)
            
            if not output_result['passed']:
                moderation_result['passed'] = False
                moderation_result['violations'].extend(output_result['violations'])
            
            # Check against policies
            policy_result = self.policy_engine.check_output_policy(
                model_output, output_result
            )
            
            if not policy_result['passed']:
                moderation_result['passed'] = False
                moderation_result['violations'].extend(policy_result['violations'])
            
        except Exception as e:
            logger.error(f"Error during output moderation: {e}", exc_info=True)
            moderation_result['passed'] = False
            moderation_result['violations'].append({
                'type': 'SYSTEM_ERROR',
                'message': str(e)
            })
        
        moderation_result['latency_ms'] = (time.time() - start_time) * 1000
        
        # Log the event
        self._log_event('output_moderation', moderation_result)
        
        return moderation_result
    

    def _log_event(self, event_type: str, result: Dict[str, Any]) -> None:
        """Log guardrail events to file and monitoring system."""
        try:
            # Log to file
            log_file = self.log_dir / f"guardrails_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            
            event = {
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'result': result
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
            
            # Log violations at appropriate level
            if not result['passed']:
                logger.warning(
                    f"Guardrail violation detected in {event_type}: "
                    f"{result['violations']}"
                )
            else:
                logger.info(f"{event_type} passed all checks")


            # Validation status
            GuardrailMetrics.record_validation(
                validation_type=event_type,
                passed=result['passed']
            )

            # Latency
            GuardrailMetrics.record_latency(
                operation=event_type,
                latency_ms=result['latency_ms']
            )

            # Violations
            for violation in result['violations']:
                GuardrailMetrics.record_violation(
                    violation_type=violation['type']
                )

            # PII detections (input only)
            if event_type == 'input_validation' and result.get('pii_detected'):
                for entity in result['pii_detected']:
                    GuardrailMetrics.record_pii_detection(
                        entity_type=entity['type']
                    )

            # Toxicity scores (output only)
            if event_type == 'output_moderation' and result.get('toxicity_scores'):
                for score in result['toxicity_scores'].values():
                    GuardrailMetrics.record_toxicity_score(score)

        except Exception as e:
            logger.error(f"Error logging guardrail event: {e}", exc_info=True)



    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about guardrail events."""
        stats = {
            'total_events': 0,
            'input_validations': 0,
            'output_moderations': 0,
            'total_violations': 0,
            'violation_types': {}
        }
        
        try:
            for log_file in self.log_dir.glob("guardrails_*.jsonl"):
                with open(log_file, 'r') as f:
                    for line in f:
                        event = json.loads(line)
                        stats['total_events'] += 1
                        
                        if event['event_type'] == 'input_validation':
                            stats['input_validations'] += 1
                        elif event['event_type'] == 'output_moderation':
                            stats['output_moderations'] += 1
                        
                        if not event['result']['passed']:
                            stats['total_violations'] += 1
                            for violation in event['result']['violations']:
                                v_type = violation.get('type', 'UNKNOWN')
                                stats['violation_types'][v_type] = \
                                    stats['violation_types'].get(v_type, 0) + 1
        
        except Exception as e:
            logger.error(f"Error calculating stats: {e}", exc_info=True)
        
        return stats
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics for monitoring."""
        stats = self.get_stats()
        
        return {
            'total_requests': stats['total_events'],
            'input_violations': sum(1 for k in stats['violation_types'] 
                                   if 'PROMPT_INJECTION' in k or 'INPUT' in k),
            'output_violations': sum(1 for k in stats['violation_types'] 
                                    if 'TOXICITY' in k or 'OUTPUT' in k),
            'pii_detections': stats['violation_types'].get('PII_DETECTED', 0),
            'prompt_injections': stats['violation_types'].get('PROMPT_INJECTION', 0),
            'toxicity_blocks': stats['violation_types'].get('TOXICITY_DETECTED', 0)
        }