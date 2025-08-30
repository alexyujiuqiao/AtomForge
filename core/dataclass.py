from abc import ABC, abstractmethod
from dataclasses import dataclass

"""
This file defines the core data structures used to represent DSL programs.

The base class `DSLProgram` is an abstract representation of a complete DSL program.
Each concrete DSL (e.g., for statistics, medicine, music, etc.) should define its own
subclass of `DSLProgram`, containing all domain-specific blocks and entities.

All DSL programs must implement a `validate` method to ensure internal consistency.
"""
@dataclass
class DSLProgram(ABC):
    @abstractmethod
    def validate(self) -> None:
        """Validate the integrity of the program"""
        pass
