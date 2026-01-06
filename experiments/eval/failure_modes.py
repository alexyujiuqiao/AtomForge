#!/usr/bin/env python3
"""
Failure Mode Taxonomy

Categorizes different types of failures in structure generation.
"""

from enum import Enum
from typing import Optional


class FailureMode(Enum):
    """Failure mode categories."""
    PARSE_ERROR = "parse_error"
    IR_VALIDATION_ERROR = "ir_validation_error"
    STRUCTURE_CONVERSION_ERROR = "structure_conversion_error"
    MIN_DIST_FAIL = "min_dist_fail"
    CHARGE_UNKNOWN = "charge_unknown"
    CHARGE_NONNEUTRAL = "charge_nonneutral"
    CONDITION_VIOLATION = "condition_violation"
    OTHER_EXCEPTION = "other_exception"
    NONE = "none"  # No failure


def categorize_failure(error_message: Optional[str], metrics: dict) -> FailureMode:
    """
    Categorize failure based on error message and metrics.
    
    Args:
        error_message: Error message string
        metrics: Dictionary of computed metrics
        
    Returns:
        FailureMode enum value
    """
    if not error_message and metrics.get("parse_ok") and metrics.get("struct_ok"):
        # Check for other failures
        if not metrics.get("valid_min_distance", True):
            return FailureMode.MIN_DIST_FAIL
        if metrics.get("charge_status") == "unknown_charge":
            return FailureMode.CHARGE_UNKNOWN
        if metrics.get("charge_status") == "charged":
            return FailureMode.CHARGE_NONNEUTRAL
        if metrics.get("condition_violation", False):
            return FailureMode.CONDITION_VIOLATION
        return FailureMode.NONE
    
    if not error_message:
        return FailureMode.OTHER_EXCEPTION
    
    error_lower = error_message.lower()
    
    # Parse errors
    if "parse" in error_lower or "syntax" in error_lower or "grammar" in error_lower:
        return FailureMode.PARSE_ERROR
    
    # IR validation errors
    if "validation" in error_lower or "validate" in error_lower:
        return FailureMode.IR_VALIDATION_ERROR
    
    # Structure conversion errors
    if "structure" in error_lower and ("convert" in error_lower or "build" in error_lower):
        return FailureMode.STRUCTURE_CONVERSION_ERROR
    
    # Condition violation
    if "constraint" in error_lower or "condition" in error_lower or "violation" in error_lower:
        return FailureMode.CONDITION_VIOLATION
    
    return FailureMode.OTHER_EXCEPTION

