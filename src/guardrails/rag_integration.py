import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

from .guardrail_engine import GuardrailEngine

logger = logging.getLogger(__name__)


class GuardrailMiddleware:
    """Middleware for integrating guardrails with RAG pipeline."""
    
    def __init__(self, guardrail_engine: GuardrailEngine):
        """
        Initialize middleware.
        
        Args:
            guardrail_engine: Instance of GuardrailEngine
        """
        self.engine = guardrail_engine
    
    def validate_and_process(
        self,
        user_query: str,
        rag_function: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate input, process through RAG, and moderate output.
        
        Args:
            user_query: User's input query
            rag_function: The RAG processing function
            metadata: Optional metadata
            
        Returns:
            Dictionary with final response and guardrail results
        """
        response = {
            'success': False,
            'query': user_query,
            'response': None,
            'input_validation': None,
            'output_moderation': None,
            'error': None
        }
        
        try:
            # Step 1: Validate input
            input_result = self.engine.validate_input(user_query, metadata)
            response['input_validation'] = input_result
            
            if not input_result['passed']:
                response['error'] = 'Input validation failed'
                response['response'] = self._generate_error_response(
                    input_result['violations']
                )
                return response
            
            # Step 2: Process through RAG with sanitized input
            sanitized_query = input_result['sanitized_input']
            rag_output = rag_function(sanitized_query)
            
            # Step 3: Moderate output
            output_result = self.engine.moderate_output(rag_output, metadata)
            response['output_moderation'] = output_result
            
            if not output_result['passed']:
                # Check if violations are blocking
                blocking_violations = [
                    v for v in output_result['violations']
                    if v.get('severity') != 'WARNING'
                ]
                
                if blocking_violations:
                    response['error'] = 'Output moderation failed'
                    response['response'] = self._generate_error_response(
                        blocking_violations
                    )
                    return response
            
            # Step 4: Return sanitized output
            response['success'] = True
            response['response'] = output_result['sanitized_output']
            
        except Exception as e:
            logger.error(f"Error in guardrail middleware: {e}", exc_info=True)
            response['error'] = str(e)
            response['response'] = "An error occurred while processing your request."
        
        return response
    
    def _generate_error_response(self, violations: list) -> str:
        """Generate user-friendly error response."""
        return (
            "I cannot process this request due to content policy violations. "
            "Please rephrase your query and try again."
        )


def with_guardrails(guardrail_engine: GuardrailEngine):
    """
    Decorator for applying guardrails to functions.
    
    Args:
        guardrail_engine: Instance of GuardrailEngine
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(query: str, *args, **kwargs) -> Dict[str, Any]:
            # Validate input
            input_result = guardrail_engine.validate_input(query)
            
            if not input_result['passed']:
                return {
                    'success': False,
                    'error': 'Input validation failed',
                    'violations': input_result['violations']
                }
            
            # Call original function
            output = func(input_result['sanitized_input'], *args, **kwargs)
            
            # Moderate output
            output_result = guardrail_engine.moderate_output(output)
            
            if not output_result['passed']:
                return {
                    'success': False,
                    'error': 'Output moderation failed',
                    'violations': output_result['violations']
                }
            
            return {
                'success': True,
                'response': output_result['sanitized_output']
            }
        
        return wrapper
    return decorator