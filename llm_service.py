import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MathEvaluationService:
    def __init__(self):
        # For demo purposes, we'll use a simple rule-based approach
        # In production, this would connect to vLLM or another LLM service
        self.evaluation_prompt_template = """
You are an expert math tutor. Your task is to evaluate a student's handwritten mathematical solution.

Student's solution: {student_solution}

Please analyze the solution and provide:
1. Whether the solution is correct (True/False)
2. Any errors found
3. Step-by-step correction if needed
4. Encouraging feedback

Respond in JSON format:
{{
    "is_correct": boolean,
    "errors": ["list of errors"],
    "correct_solution": "step-by-step correct solution",
    "feedback": "encouraging feedback for the student"
}}
"""

    def evaluate_solution(self, student_text: str) -> Dict[str, Any]:
        """
        Evaluate a student's mathematical solution
        For now, this is a simplified version. In production, this would call vLLM.
        """
        try:
            # Simple pattern matching for demo
            result = self._simple_evaluation(student_text)
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "is_correct": False,
                "errors": [f"Evaluation error: {str(e)}"],
                "correct_solution": "Unable to evaluate at this time",
                "feedback": "Please try again or check your handwriting clarity"
            }

    def _simple_evaluation(self, text: str) -> Dict[str, Any]:
        """
        Simplified evaluation logic for demonstration
        In production, this would be replaced with LLM inference
        """
        text_lower = text.lower().strip()
        
        # Basic math problem detection and evaluation
        if any(op in text for op in ['+', '-', '*', '/', '=', 'integral', '∫', 'derivative']):
            # Check for common patterns
            if '=' in text:
                parts = text.split('=')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Simple arithmetic check
                    try:
                        # Very basic evaluation for simple expressions
                        if self._is_simple_arithmetic(left, right):
                            is_correct = self._check_arithmetic(left, right)
                            return {
                                "is_correct": is_correct,
                                "errors": [] if is_correct else ["Arithmetic error detected"],
                                "correct_solution": f"Please verify: {left} = {right}",
                                "feedback": "Good work on showing your steps!" if is_correct else "Check your arithmetic - you're on the right track!"
                            }
                    except:
                        pass
            
            # For calculus problems
            if any(word in text_lower for word in ['integral', 'derivative', '∫', 'dx', 'dy']):
                return {
                    "is_correct": True,  # Assume correct for demo
                    "errors": [],
                    "correct_solution": "Calculus problem detected - great work on tackling advanced math!",
                    "feedback": "Excellent effort on this calculus problem! Keep practicing these techniques."
                }
        
        # Default response
        return {
            "is_correct": None,
            "errors": ["Unable to parse mathematical content"],
            "correct_solution": "Please ensure your mathematical notation is clear",
            "feedback": "I can see you've written something mathematical. Try making your handwriting clearer for better analysis."
        }

    def _is_simple_arithmetic(self, left: str, right: str) -> bool:
        """Check if this is a simple arithmetic expression"""
        try:
            # Very basic check for simple numbers and operations
            import re
            pattern = r'^[\d\+\-\*/\.\s\(\)]+$'
            return bool(re.match(pattern, left)) and bool(re.match(pattern, right))
        except:
            return False

    def _check_arithmetic(self, left: str, right: str) -> bool:
        """Very basic arithmetic verification"""
        try:
            # WARNING: In production, never use eval() on user input!
            # This is just for demo purposes with controlled input
            left_val = eval(left.replace(' ', ''))
            right_val = float(right.replace(' ', ''))
            return abs(left_val - right_val) < 0.001
        except:
            return False

# Global service instance
math_eval_service = None

def get_math_evaluation_service():
    global math_eval_service
    if math_eval_service is None:
        math_eval_service = MathEvaluationService()
    return math_eval_service