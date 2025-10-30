"""
Memory helper for extracting and storing important information.
"""

import re
from typing import Dict, Any, Optional


class MemoryHelper:
    """Helper class for managing agent memory extraction."""
    
    @staticmethod
    def extract_user_name(text: str) -> Optional[str]:
        """Extract user's name from conversation."""
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1)
                # Capitalize first letter
                return name.capitalize()
        
        return None
    
    @staticmethod
    def should_remember(text: str) -> bool:
        """Check if text contains information worth remembering."""
        remember_keywords = [
            "my name",
            "i am",
            "i'm",
            "call me",
            "remember",
            "don't forget",
            "i like",
            "i love",
            "i hate",
            "i prefer",
            "my favorite",
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in remember_keywords)
    
    @staticmethod
    def auto_extract_facts(user_message: str, assistant_response: str = "") -> Dict[str, Any]:
        """Automatically extract facts from conversation."""
        facts = {}
        
        # Try to extract name
        name = MemoryHelper.extract_user_name(user_message)
        if name:
            facts["user_name"] = name
        
        # Could add more extraction patterns here:
        # - Location
        # - Age
        # - Preferences
        # - Occupation
        # etc.
        
        return facts
