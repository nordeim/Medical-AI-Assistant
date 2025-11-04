"""
Agent Configuration

Configuration dataclass for the medical AI agent.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for the medical AI agent"""
    
    # System Prompts
    system_prompt_path: Path = Path("app/prompts/system_prompt.txt")
    par_prompt_path: Path = Path("app/prompts/par_generation_prompt.txt")
    safety_prompt_path: Path = Path("app/prompts/safety_prompt.txt")
    
    # Model Parameters
    model_path: str = ""
    device: str = "cuda"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    use_8bit: bool = True
    use_flash_attention: bool = False
    
    # Agent Parameters
    max_iterations: int = 10
    stream_tokens: bool = True
    verbose: bool = False
    
    # RAG Parameters
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranker: bool = True
    
    # Safety Parameters
    enable_safety_filters: bool = True
    block_diagnosis: bool = True
    block_prescription: bool = True
    
    # Logging
    log_tool_calls: bool = True
    log_token_usage: bool = True
    
    def load_system_prompt(self) -> str:
        """Load system prompt from file"""
        if self.system_prompt_path.exists():
            return self.system_prompt_path.read_text()
        return "You are a helpful medical triage assistant."
    
    def load_par_prompt(self) -> str:
        """Load PAR generation prompt from file"""
        if self.par_prompt_path.exists():
            return self.par_prompt_path.read_text()
        return "Generate a preliminary assessment report."
    
    def load_safety_prompt(self) -> str:
        """Load safety prompt from file"""
        if self.safety_prompt_path.exists():
            return self.safety_prompt_path.read_text()
        return "Ensure responses are safe and appropriate."
    
    @classmethod
    def from_settings(cls, settings):
        """Create AgentConfig from application settings"""
        return cls(
            model_path=settings.MODEL_PATH,
            device=settings.MODEL_DEVICE,
            max_length=settings.MODEL_MAX_LENGTH,
            temperature=settings.MODEL_TEMPERATURE,
            top_p=settings.MODEL_TOP_P,
            use_8bit=settings.USE_8BIT,
            use_flash_attention=settings.USE_FLASH_ATTENTION,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            stream_tokens=settings.AGENT_STREAM_TOKENS,
            verbose=settings.AGENT_VERBOSE,
            top_k=settings.RAG_TOP_K,
            similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD,
            use_reranker=settings.RAG_USE_RERANKER,
            enable_safety_filters=settings.SAFETY_ENABLED,
            block_diagnosis=settings.SAFETY_BLOCK_DIAGNOSIS,
            block_prescription=settings.SAFETY_BLOCK_PRESCRIPTION,
        )
