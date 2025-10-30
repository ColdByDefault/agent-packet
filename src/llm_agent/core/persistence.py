"""
Conversation persistence utilities.
Saves and loads conversation history to/from disk.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class ConversationPersistence:
    """Handles saving and loading conversation history with session support."""
    
    def __init__(self, storage_path: str = "./data/conversations"):
        """Initialize persistence with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_file = self.storage_path / "current_conversation.json"
        self.sessions_dir = self.storage_path / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.memory_file = self.storage_path / "agent_memory.json"
        self.current_session_id: Optional[str] = None
    
    def save_conversation(self, messages: List[Dict[str, Any]], session_id: Optional[str] = None) -> str:
        """Save conversation to disk. Returns session_id."""
        try:
            if session_id is None:
                session_id = self.current_session_id or str(uuid.uuid4())
            
            self.current_session_id = session_id
            
            data = {
                "session_id": session_id,
                "messages": messages,
                "saved_at": datetime.now().isoformat(),
                "message_count": len(messages)
            }
            
            # Save to current file
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Also save to session file
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return session_id
                
        except Exception as e:
            print(f"Warning: Failed to save conversation: {e}")
            return session_id or str(uuid.uuid4())
    
    def load_conversation(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load conversation from disk."""
        try:
            if session_id:
                # Load specific session
                session_file = self.sessions_dir / f"{session_id}.json"
                if not session_file.exists():
                    return []
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_session_id = session_id
                    return data.get("messages", [])
            else:
                # Load current conversation
                if not self.current_file.exists():
                    return []
                
                with open(self.current_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_session_id = data.get("session_id")
                    return data.get("messages", [])
                
        except Exception as e:
            print(f"Warning: Failed to load conversation: {e}")
            return []
    
    def save_memory(self, memory_data: Dict[str, Any]) -> None:
        """Save agent's long-term memory."""
        try:
            data = {
                "memory": memory_data,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save memory: {e}")
    
    def load_memory(self) -> Dict[str, Any]:
        """Load agent's long-term memory."""
        try:
            if not self.memory_file.exists():
                return {}
            
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("memory", {})
        except Exception as e:
            print(f"Warning: Failed to load memory: {e}")
            return {}
    
    def start_new_session(self) -> str:
        """Start a new conversation session. Returns new session_id."""
        new_session_id = str(uuid.uuid4())
        self.current_session_id = new_session_id
        
        # Save empty conversation for new session
        self.save_conversation([], new_session_id)
        
        return new_session_id
    
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
