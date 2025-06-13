"""
Memory manager for APU conversations
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

from config.settings import DATA_DIR
from utils.logger import logger

class MemoryManager:
    """Manages conversation memory and sessions"""
    
    def __init__(self):
        self.sessions_dir = DATA_DIR / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.sessions = {}
        logger.info("Memory manager initialized")
    
    def create_session(self) -> str:
        """Creates a new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": {}
        }
        logger.info(f"New session created: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Adds a message to a session"""
        if session_id not in self.sessions:
            session_id = self.create_session()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.sessions[session_id]["messages"].append(message)
        logger.debug(f"Message added to session {session_id}: {role}")
        return message
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Gets a session by ID"""
        return self.sessions.get(session_id)
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Gets messages from a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session["messages"]
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def export_session(self, session_id: str, format_type: str = "json") -> str:
        """Exports a session to file"""
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            filepath = self.sessions_dir / f"session_{timestamp}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
            logger.info(f"Session exported to JSON: {filepath}")
        
        elif format_type == "markdown":
            filepath = self.sessions_dir / f"session_{timestamp}.md"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Session Export - {timestamp}\n\n")
                f.write(f"**Session ID**: {session_id}\n")
                f.write(f"**Created**: {session['created_at']}\n")
                f.write(f"**Messages**: {len(session['messages'])}\n\n")
                
                for i, msg in enumerate(session["messages"], 1):
                    role = "User" if msg["role"] == "user" else "APU"
                    timestamp = msg.get("timestamp", "")
                    content = msg.get("content", "")
                    
                    f.write(f"## Message {i} - {role}\n")
                    if timestamp:
                        f.write(f"*{timestamp}*\n\n")
                    f.write(f"{content}\n\n")
                    f.write("---\n\n")
            
            logger.info(f"Session exported to Markdown: {filepath}")
        
        return str(filepath)
    
    def clear_session(self, session_id: str):
        """Clears a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"] = []
            logger.info(f"Session cleared: {session_id}")
    
    def delete_session(self, session_id: str):
        """Deletes a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Gets statistics for a session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        messages = session["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        
        # Calculate duration
        if messages:
            start_time = datetime.fromisoformat(messages[0]["timestamp"])
            end_time = datetime.fromisoformat(messages[-1]["timestamp"])
            duration = (end_time - start_time).total_seconds() / 60  # minutes
        else:
            duration = 0
        
        # Extract tools used
        tools_used = set()
        documents_accessed = set()
        
        for msg in assistant_messages:
            metadata = msg.get("metadata", {})
            if "tool_used" in metadata:
                tools_used.add(metadata["tool_used"])
            if "documents_accessed" in metadata:
                documents_accessed.update(metadata["documents_accessed"])
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "duration_minutes": round(duration, 2),
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "tools_usage": {
                "search_documents": sum(1 for m in assistant_messages 
                                     if m.get("metadata", {}).get("tool_used") == "search_documents"),
                "web_search": sum(1 for m in assistant_messages 
                                if m.get("metadata", {}).get("tool_used") == "web_search")
            },
            "documents_accessed": len(documents_accessed),
            "unique_tools": list(tools_used)
        }
    
    def save_to_disk(self, session_id: str):
        """Saves a session to disk"""
        session = self.get_session(session_id)
        if not session:
            return
        
        filepath = self.sessions_dir / f"{session_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session saved to disk: {filepath}")
    
    def load_from_disk(self, session_id: str) -> bool:
        """Loads a session from disk"""
        filepath = self.sessions_dir / f"{session_id}.json"
        
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session = json.load(f)
            
            self.sessions[session_id] = session
            logger.info(f"Session loaded from disk: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session from disk: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all active sessions"""
        sessions_info = []
        
        for session_id, session in self.sessions.items():
            info = {
                "id": session_id,
                "created_at": session["created_at"],
                "message_count": len(session["messages"]),
                "last_activity": session["messages"][-1]["timestamp"] if session["messages"] else session["created_at"]
            }
            sessions_info.append(info)
        
        # Sort by last activity
        sessions_info.sort(key=lambda x: x["last_activity"], reverse=True)
        
        return sessions_info