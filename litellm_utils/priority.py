from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from litellm_utils.handler import request_ai
from litellm_utils.types import FileType

class ModelPriority(BaseModel):
    model: str
    budget: int
    priority: int

class LiteLLMUtils:
    def __init__(self, priorities: list[ModelPriority]):
        self.priorities: list[ModelPriority] = priorities
        self._tokens_used: dict = {priority.model: 0 for priority in priorities}

    def _get_current_model(self):
        for priority in sorted(self.priorities, key = lambda p: p.priority):
            if self._tokens_used[priority.model] < priority.budget:
                return priority.model
        return None

    def request_ai(
        self,
        system_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
        messages: list[dict] = None,
        file: Optional[FileType | list[FileType]] = None,
        temperature: float = 0.2,
        preprocess_file_content: Optional[bool] = None,
        json_output: bool = False
    ):
        request_ai(
            model=self._get_current_model(),
            system_prompt=system_prompt,
            user_text=user_text,
            messages=messages,
            file=file,
            temperature=temperature,
            preprocess_file_content=preprocess_file_content,
            json_output=json_output,
        )
