#!/usr/bin/env python3
"""
Common utilities for baseline runners.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class GenerationAttempt:
    """A single generation attempt with metadata."""
    attempt_num: int
    prompt: str
    response: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error: Optional[str] = None
    success: bool = False
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class LLMClient:
    """LLM client interface (OpenAI-compatible)."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = kwargs.get('temperature', 0.8)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                self.available = True
            except Exception:
                self.available = False
                self.client = None
        else:
            self.available = False
            self.client = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from prompt.
        
        Returns:
            (response_text, metadata) where metadata includes token counts if available
        """
        if not self.available or not self.client:
            # Stub mode: return a placeholder
            return (
                "# STUB MODE: No API key available\n# This is a placeholder response.",
                {"tokens_in": len(prompt) // 4, "tokens_out": 100, "model": self.model}
            )
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            
            text = response.choices[0].message.content
            metadata = {
                "tokens_in": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
                "tokens_out": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None,
                "model": self.model
            }
            return text, metadata
        except Exception as e:
            return f"# ERROR: {str(e)}", {"error": str(e)}


def save_artifacts(
    output_dir: Path,
    task_id: str,
    attempts: List[GenerationAttempt],
    final_programs: List[Tuple[str, str]],  # (filename, content)
    metadata: Dict[str, Any]
) -> None:
    """Save generation artifacts to disk."""
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    programs_dir = task_dir / "programs"
    programs_dir.mkdir(exist_ok=True)
    
    logs_dir = task_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Save programs
    for filename, content in final_programs:
        (programs_dir / filename).write_text(content, encoding='utf-8')
    
    # Save attempts
    attempts_file = logs_dir / "attempts.jsonl"
    with open(attempts_file, 'w', encoding='utf-8') as f:
        for attempt in attempts:
            f.write(json.dumps(asdict(attempt)) + '\n')
    
    # Save prompts/responses
    if attempts:
        (logs_dir / "prompt.txt").write_text(attempts[0].prompt, encoding='utf-8')
        (logs_dir / "response.txt").write_text(attempts[-1].response, encoding='utf-8')
    
    # Save metadata
    (task_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding='utf-8'
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4

