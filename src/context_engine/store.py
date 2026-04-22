from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from context_engine.schemas import MemoryItem, SessionState, UserProfile
from legal_agent.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


class DiskMemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(root)

    def users_root(self) -> Path:
        return ensure_dir(self.root / "users")

    def user_dir(self, user_id: str) -> Path:
        return ensure_dir(self.root / "users" / user_id)

    def sessions_dir(self, user_id: str) -> Path:
        return ensure_dir(self.user_dir(user_id) / "sessions")

    def profile_path(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "profile.json"

    def memories_path(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "memories.jsonl"

    def session_path(self, user_id: str, session_id: str) -> Path:
        return self.sessions_dir(user_id) / f"{session_id}.json"

    def system_memories_path(self) -> Path:
        return self.root / "system" / "system_memories.jsonl"

    def list_users(self) -> list[str]:
        users_root = self.root / "users"
        if not users_root.exists():
            return []
        return sorted(path.name for path in users_root.iterdir() if path.is_dir())

    def user_exists(self, user_id: str) -> bool:
        return (self.root / "users" / user_id).is_dir()

    def delete_user(self, user_id: str) -> bool:
        user_dir = self.root / "users" / user_id
        if not user_dir.exists():
            return False
        shutil.rmtree(user_dir)
        return True

    def list_sessions(self, user_id: str) -> list[str]:
        sessions_dir = self.root / "users" / user_id / "sessions"
        if not sessions_dir.exists():
            return []
        return sorted(path.stem for path in sessions_dir.glob("*.json"))

    def session_exists(self, user_id: str, session_id: str) -> bool:
        return self.session_path(user_id, session_id).exists()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        path = self.session_path(user_id, session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def create_user(self, user_id: str, *, default_name: str | None = None) -> UserProfile:
        profile = self.load_profile(user_id)
        if default_name and not profile.name:
            profile.name = default_name
        self.save_profile(profile)
        return profile

    def create_session(self, user_id: str, session_id: str) -> SessionState:
        state = self.load_session_state(user_id, session_id)
        self.save_session_state(state)
        return state

    def load_profile(self, user_id: str) -> UserProfile:
        path = self.profile_path(user_id)
        if not path.exists():
            return UserProfile(user_id=user_id)
        return UserProfile.from_dict(read_json(path))

    def save_profile(self, profile: UserProfile) -> None:
        write_json(self.profile_path(profile.user_id), profile.to_dict())

    def load_user_memories(self, user_id: str) -> list[MemoryItem]:
        path = self.memories_path(user_id)
        if not path.exists():
            return []
        return [MemoryItem.from_dict(row) for row in read_jsonl(path)]

    def save_user_memories(self, user_id: str, memories: list[MemoryItem]) -> None:
        write_jsonl(self.memories_path(user_id), [item.to_dict() for item in memories])

    def load_system_memories(self) -> list[MemoryItem]:
        path = self.system_memories_path()
        if not path.exists():
            return []
        return [MemoryItem.from_dict(row) for row in read_jsonl(path)]

    def save_system_memories(self, memories: list[MemoryItem]) -> None:
        write_jsonl(self.system_memories_path(), [item.to_dict() for item in memories])

    def _recover_session_state(self, path: Path, user_id: str, session_id: str) -> SessionState:
        if path.exists():
            backup_path = path.with_name(f"{path.stem}.corrupt-{uuid4().hex[:8]}{path.suffix}")
            path.replace(backup_path)

        state = SessionState(session_id=session_id, user_id=user_id)
        self.save_session_state(state)
        return state

    def load_session_state(self, user_id: str, session_id: str) -> SessionState:
        path = self.session_path(user_id, session_id)
        if not path.exists():
            return SessionState(session_id=session_id, user_id=user_id)
        try:
            return SessionState.from_dict(read_json(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return self._recover_session_state(path, user_id, session_id)

    def save_session_state(self, state: SessionState) -> None:
        write_json(self.session_path(state.user_id, state.session_id), state.to_dict())