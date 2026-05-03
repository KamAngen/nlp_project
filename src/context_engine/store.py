from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

from context_engine.schemas import ConversationTurn, MemoryEdge, MemoryItem, SessionState, UserProfile
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

    def memory_edges_path(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "memory_edges.jsonl"

    def session_dir(self, user_id: str, session_id: str) -> Path:
        return self.sessions_dir(user_id) / session_id

    def session_path(self, user_id: str, session_id: str) -> Path:
        return self.sessions_dir(user_id) / f"{session_id}.json"

    def legacy_session_path(self, user_id: str, session_id: str) -> Path:
        return self.session_dir(user_id, session_id) / "state.json"

    def session_turns_path(self, user_id: str, session_id: str) -> Path:
        return self.session_dir(user_id, session_id) / "turns.jsonl"

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
        session_ids: set[str] = set()
        for path in sessions_dir.iterdir():
            if path.is_dir() and ((path / "state.json").exists() or (path / "turns.jsonl").exists()):
                session_ids.add(path.name)
            elif path.is_file() and path.suffix == ".json":
                session_ids.add(path.stem)
        return sorted(session_ids)

    def session_exists(self, user_id: str, session_id: str) -> bool:
        return self.session_path(user_id, session_id).exists() or self.legacy_session_path(user_id, session_id).exists()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        removed = False
        session_dir = self.session_dir(user_id, session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            removed = True
        session_path = self.session_path(user_id, session_id)
        if session_path.exists():
            session_path.unlink()
            removed = True
        return removed

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

    def load_memory_edges(self, user_id: str) -> list[MemoryEdge]:
        path = self.memory_edges_path(user_id)
        if not path.exists():
            return []
        return [MemoryEdge.from_dict(row) for row in read_jsonl(path)]

    def save_memory_edges(self, user_id: str, edges: list[MemoryEdge]) -> None:
        path = self.memory_edges_path(user_id)
        if not edges:
            if path.exists():
                path.unlink()
            return
        write_jsonl(path, [edge.to_dict() for edge in edges])

    def load_system_memories(self) -> list[MemoryItem]:
        path = self.system_memories_path()
        if not path.exists():
            return []
        return [MemoryItem.from_dict(row) for row in read_jsonl(path)]

    def save_system_memories(self, memories: list[MemoryItem]) -> None:
        write_jsonl(self.system_memories_path(), [item.to_dict() for item in memories])

    def _recover_session_state(self, path: Path, user_id: str, session_id: str) -> SessionState:
        if path.exists():
            backup_path = path.parent / f"{session_id}.corrupt-{uuid4().hex[:8]}{path.suffix}"
            path.replace(backup_path)

        state = SessionState(session_id=session_id, user_id=user_id)
        self.save_session_state(state)
        return state

    def _load_session_file(self, path: Path) -> SessionState:
        return SessionState.from_dict(read_json(path))

    def _load_legacy_session_state(self, user_id: str, session_id: str) -> SessionState:
        state_payload = read_json(self.legacy_session_path(user_id, session_id))
        state = SessionState.from_dict(state_payload)
        turns_path = self.session_turns_path(user_id, session_id)
        if turns_path.exists():
            turns = [ConversationTurn.from_dict(row) for row in read_jsonl(turns_path)]
        else:
            turns = list(state.turns)
        state.turns = turns
        state.turn_count = max(int(state_payload.get("turn_count") or 0), len(turns))
        state.last_turn_id = turns[-1].turn_id if turns else None
        self.save_session_state(state)
        self._cleanup_legacy_session_dir(user_id, session_id)
        return state

    def _cleanup_legacy_session_dir(self, user_id: str, session_id: str) -> None:
        session_dir = self.session_dir(user_id, session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def _existing_session_turns(self, user_id: str, session_id: str) -> list[ConversationTurn]:
        path = self.session_path(user_id, session_id)
        if path.exists():
            try:
                return self._load_session_file(path).turns
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return []
        if self.legacy_session_path(user_id, session_id).exists():
            try:
                return self._load_legacy_session_state(user_id, session_id).turns
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return []
        return []

    def _merge_turns_for_save(
        self,
        state: SessionState,
        existing_turns: list[ConversationTurn],
    ) -> list[ConversationTurn]:
        incoming_turns = list(state.turns)
        if not incoming_turns:
            return list(existing_turns)
        if not existing_turns:
            return incoming_turns
        if state.turn_count > len(incoming_turns) and len(existing_turns) >= state.turn_count:
            return list(existing_turns)
        if len(incoming_turns) < len(existing_turns):
            suffix_ids = [turn.turn_id for turn in incoming_turns]
            existing_suffix = [turn.turn_id for turn in existing_turns[-len(incoming_turns) :]] if incoming_turns else []
            if suffix_ids and suffix_ids == existing_suffix:
                return list(existing_turns)
            if state.last_turn_id and existing_turns[-1].turn_id == state.last_turn_id:
                return list(existing_turns)
        return incoming_turns

    def load_session_state(self, user_id: str, session_id: str) -> SessionState:
        path = self.session_path(user_id, session_id)
        legacy_path = self.legacy_session_path(user_id, session_id)
        if not path.exists() and not legacy_path.exists():
            return SessionState(session_id=session_id, user_id=user_id)
        try:
            if path.exists():
                return self._load_session_file(path)
            return self._load_legacy_session_state(user_id, session_id)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            recover_path = path if path.exists() else legacy_path
            return self._recover_session_state(recover_path, user_id, session_id)

    def save_session_state(self, state: SessionState) -> None:
        path = self.session_path(state.user_id, state.session_id)
        existing_turns = self._existing_session_turns(state.user_id, state.session_id)
        merged_turns = self._merge_turns_for_save(state, existing_turns)
        payload = state.to_dict()
        payload["turns"] = [turn.to_dict() for turn in merged_turns]
        payload["turn_count"] = max(int(payload.get("turn_count") or 0), len(merged_turns))
        payload["last_turn_id"] = merged_turns[-1].turn_id if merged_turns else payload.get("last_turn_id")
        write_json(path, payload)
        self._cleanup_legacy_session_dir(state.user_id, state.session_id)

    def load_session_turns(self, user_id: str, session_id: str, *, limit: int | None = None) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        try:
            path = self.session_path(user_id, session_id)
            if path.exists():
                turns = self._load_session_file(path).turns
            elif self.legacy_session_path(user_id, session_id).exists():
                turns = self._load_legacy_session_state(user_id, session_id).turns
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            recover_path = self.session_path(user_id, session_id)
            if not recover_path.exists():
                recover_path = self.session_turns_path(user_id, session_id)
            if recover_path.exists():
                backup_path = recover_path.parent / f"{session_id}.corrupt-{uuid4().hex[:8]}{recover_path.suffix}"
                recover_path.replace(backup_path)
            turns = []
        if limit is None:
            return turns
        return turns[-limit:]

    def save_session_turns(self, user_id: str, session_id: str, turns: list[ConversationTurn]) -> None:
        state = self.load_session_state(user_id, session_id)
        state.turns = list(turns)
        state.turn_count = len(turns)
        state.last_turn_id = turns[-1].turn_id if turns else None
        self.save_session_state(state)

    def append_session_turn(self, user_id: str, session_id: str, turn: ConversationTurn) -> None:
        state = self.load_session_state(user_id, session_id)
        turns = list(state.turns)
        turns.append(turn)
        state.turns = turns
        state.turn_count = len(turns)
        state.last_turn_id = turn.turn_id
        self.save_session_state(state)

    def replace_last_session_turn(self, user_id: str, session_id: str, turn: ConversationTurn) -> None:
        turns = self.load_session_turns(user_id, session_id)
        if not turns:
            return
        turns[-1] = turn
        self.save_session_turns(user_id, session_id, turns)