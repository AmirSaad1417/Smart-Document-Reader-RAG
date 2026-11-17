from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ConversationMemory:
    """Stores and formats conversational interaction history."""

    max_turns: int = 10
    messages: List[Tuple[str, str]] = field(default_factory=list)

    def add_interaction(self, human_message: str, ai_response: str) -> None:
        """Appends a new turn and trims older messages beyond max_turns."""
        self.messages.append((human_message, ai_response))
        if len(self.messages) > self.max_turns:
            self.messages = self.messages[-self.max_turns:]

    def get_formatted_history(self) -> str:
        """Formats stored turns into a cohesive dialog string for the prompt."""
        if not self.messages:
            return "None"
        formatted_turns = [
            f"Human: {human}\nAssistant: {ai}"
            for human, ai in self.messages
        ]
        return "\n\n".join(formatted_turns)

    def clear(self) -> None:
        """Clears all stored conversation history."""
        self.messages.clear()
