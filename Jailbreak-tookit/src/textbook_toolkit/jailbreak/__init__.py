"""越狱章节的提示构造子包，与未来 alignment / unlearning 子包并列。"""

from .artifacts import Fragment, FragmentProvider, OptimizedPrompt, Suffix
from .base import Assembler, Pipeline
from .catalog import describe, get, list_methods, register
from .generation import AutoDAN, RedTeaming, sentence_crossover
from .manual import Cipher, Context, RolePlay, RuleRewrite, TaskDisguise, Template
from .types import Message, Prompt, Step, TextGenerator

__all__ = [
    "Assembler",
    "AutoDAN",
    "Cipher",
    "Context",
    "Fragment",
    "FragmentProvider",
    "Message",
    "OptimizedPrompt",
    "Pipeline",
    "Prompt",
    "RedTeaming",
    "RolePlay",
    "RuleRewrite",
    "Step",
    "Suffix",
    "TaskDisguise",
    "Template",
    "TextGenerator",
    "describe",
    "get",
    "list_methods",
    "register",
    "sentence_crossover",
]
