"""
Conversation persistence utilities.
Saves and loads conversation history to/from disk.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ConversationPersistence:
    """Handles saving and loading conversation history."""
    
    def __init__(self, storage_path: str = "./data/conversations"):
        """Initialize persistence with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_file = self.storage_path / "current_conversation.json"
    
    def save_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """Save conversation to disk."""
        try:
            data = {
                "messages": messages,
                "saved_at": datetime.now().isoformat(),
                "message_count": len(messages)
            }
            
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Failed to save conversation: {e}")
    
    def load_conversation(self) -> List[Dict[str, Any]]:
        """Load conversation from disk."""
        try:
            if not self.current_file.exists():
                return []
            
            with open(self.current_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("messages", [])
                
        except Exception as e:
            print(f"Warning: Failed to load conversation: {e}")
            return []
    
    def clear_conversation(self) -> None:
        """Clear saved conversation."""
        try:
            if self.current_file.exists():
                self.current_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear conversation: {e}")
    
    def export_conversation(self, filename: Optional[str] = None) -> str:
        """Export conversation to a timestamped file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"
            
            export_path = self.storage_path / filename
            
            if self.current_file.exists():
                with open(self.current_file, 'r') as src:
                    data = json.load(src)
                
                with open(export_path, 'w', encoding='utf-8') as dst:
                    json.dump(data, dst, indent=2, ensure_ascii=False)
                
                return str(export_path)
            
            return ""
            
        except Exception as e:
            print(f"Warning: Failed to export conversation: {e}")
            return ""
