# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF mode user queue and session time limit."""

import math
import threading
import time
from collections.abc import Callable
from typing import Any

import viser

from .config import DEMO_UI_QUICK_START_MODAL_MD, MAX_SESSION_MINUTES

# Link for "Duplicate this Space" on Hugging Face (used in queue and expiry modals).
DUPLICATE_SPACE_URL = "https://huggingface.co/spaces/nvidia/Kimodo?duplicate=true"
GITHUB_REPO_URL = "https://github.com/nv-tlabs/kimodo"

# How often to refresh queue modal content (position, total, estimated wait).
QUEUE_MODAL_REFRESH_INTERVAL_SEC = 15


class UserQueue:
    """Thread-safe queue: active users (with activation timestamp) and waiting queue."""

    def __init__(self, max_active: int, max_minutes: float) -> None:
        self._max_active = max_active
        self._max_minutes = max_minutes
        self._max_seconds = max_minutes * 60.0
        self._active: dict[int, float] = {}  # client_id -> activation timestamp
        self._queued: list[int] = []
        self._lock = threading.Lock()

    def try_activate(self, client_id: int) -> bool:
        """If a slot is free, add client as active and return True.

        Else return False.
        """
        with self._lock:
            if len(self._active) < self._max_active:
                self._active[client_id] = time.time()
                return True
            return False

    def enqueue(self, client_id: int) -> None:
        with self._lock:
            if client_id not in self._queued:
                self._queued.append(client_id)

    def remove(self, client_id: int) -> bool:
        """Remove from active or queue.

        Returns True if was active.
        """
        with self._lock:
            was_active = client_id in self._active
            self._active.pop(client_id, None)
            if client_id in self._queued:
                self._queued.remove(client_id)
            return was_active

    def promote_next(self) -> int | None:
        """If queue non-empty, pop first, activate them, return their client_id.

        Else None.
        """
        with self._lock:
            if not self._queued:
                return None
            client_id = self._queued.pop(0)
            self._active[client_id] = time.time()
            return client_id

    def get_queue_position(self, client_id: int) -> tuple[int, int] | None:
        """(1-based position, total_in_queue) or None if not queued."""
        with self._lock:
            if client_id not in self._queued:
                return None
            pos = self._queued.index(client_id)
            return (pos + 1, len(self._queued))

    def get_estimated_wait_seconds(self, client_id: int) -> float:
        """Estimated seconds until this queued client gets a slot."""
        with self._lock:
            if client_id not in self._queued:
                return 0.0
            pos = self._queued.index(client_id) + 1  # 1-based
            # Expiry times of active users (when they free a slot)
            now = time.time()
            expiries = sorted(now + self._max_seconds - (now - t) for t in self._active.values())
            if not expiries:
                return 0.0
            # Nth slot to free (1-indexed) wraps over expiries
            idx = (pos - 1) % len(expiries)
            cycles = (pos - 1) // len(expiries)
            slot_free_time = expiries[idx] + cycles * self._max_seconds
            return max(0.0, slot_free_time - now)

    def is_active(self, client_id: int) -> bool:
        with self._lock:
            return client_id in self._active

    def was_active(self, client_id: int) -> bool:
        """True if client is currently active (for use when already holding lock)."""
        return client_id in self._active


def _format_wait(seconds: float) -> str:
    if seconds < 60:
        return "less than a minute"
    mins = int(math.ceil(seconds / 60))
    return f"~{mins} minute{'s' if mins != 1 else ''}"


def _queue_modal_markdown(position: int, total: int, estimated_wait_sec: float) -> str:
    wait_str = _format_wait(estimated_wait_sec)
    mins = int(MAX_SESSION_MINUTES) if MAX_SESSION_MINUTES == int(MAX_SESSION_MINUTES) else MAX_SESSION_MINUTES
    return f"""## Kimodo Demo — Please Wait

This demo runs with limited capacity.
Each user gets **{mins} minute{"s" if mins != 1 else ""}** of interactive time.

**Your position in queue:** {position} / {total}

**Estimated wait:** {wait_str}

Please keep this tab open — the demo will start automatically when it's your turn.

---
*Want unlimited access? [Duplicate this Space]({DUPLICATE_SPACE_URL}) or clone the [GitHub repo]({GITHUB_REPO_URL}) to run locally!*
"""


def _welcome_modal_markdown() -> str:
    mins = int(MAX_SESSION_MINUTES) if MAX_SESSION_MINUTES == int(MAX_SESSION_MINUTES) else MAX_SESSION_MINUTES
    return f"""## Welcome to Kimodo Demo

You have been granted a **{mins}-minute** demo session.
Your session timer has started.

Click the button below to begin!
"""


def _expiry_modal_markdown() -> str:
    mins = int(MAX_SESSION_MINUTES) if MAX_SESSION_MINUTES == int(MAX_SESSION_MINUTES) else MAX_SESSION_MINUTES
    return f"""## Session Expired

Your {mins}-minute demo session has ended.
Thank you for trying Kimodo!

Refresh this page to rejoin the queue, or [duplicate this Space]({DUPLICATE_SPACE_URL}) for unlimited access.
"""


class QueueManager:
    """Orchestrates HF mode: queue modals, welcome modal, session timer, promotion."""

    def __init__(
        self,
        queue: UserQueue,
        server: viser.ViserServer,
        setup_demo_for_client: Callable[[viser.ClientHandle], None],
        cleanup_session: Callable[[int], None],
    ) -> None:
        self._queue = queue
        self._server = server
        self._setup_demo_for_client = setup_demo_for_client
        self._cleanup_session = cleanup_session
        self._max_seconds = queue._max_seconds

        self._queue_modal_handles: dict[int, tuple[Any, Any]] = {}
        self._welcome_modal_handles: dict[int, Any] = {}
        self._expiry_timers: dict[int, threading.Timer] = {}
        self._lock = threading.Lock()
        self._refresh_stop = threading.Event()
        self._refresh_thread = threading.Thread(
            target=self._queue_modal_refresh_loop,
            name="queue-modal-refresh",
            daemon=True,
        )
        self._refresh_thread.start()

    def _queue_modal_refresh_loop(self) -> None:
        """Periodically refresh queue modals so position, total, and estimated wait stay current."""
        while not self._refresh_stop.wait(timeout=QUEUE_MODAL_REFRESH_INTERVAL_SEC):
            self._update_all_queue_modals()

    def on_client_connect(self, client: viser.ClientHandle) -> None:
        """Handle new connection: activate if slot free, else enqueue and show queue modal."""
        client_id = client.client_id
        if self._queue.try_activate(client_id):
            try:
                self._setup_demo_for_client(client)
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"CUDA error while setting up client {client_id}: {e}")
                    return
                raise
            self._start_session_timer(client_id)
            self._show_welcome_modal(client)
        else:
            self._queue.enqueue(client_id)
            self._show_queue_modal(client)
            self._update_all_queue_modals()

    def on_client_disconnect(self, client_id: int) -> None:
        """Remove from queue/active, cancel timer, promote next if was active.

        Session/scene cleanup is done by the demo's on_client_disconnect.
        """
        with self._lock:
            self._expiry_timers.pop(client_id, None)
            self._queue_modal_handles.pop(client_id, None)
            self._welcome_modal_handles.pop(client_id, None)
        was_active = self._queue.remove(client_id)
        if was_active:
            self._promote_next_user()
        else:
            self._update_all_queue_modals()

    def _show_queue_modal(self, client: viser.ClientHandle) -> None:
        client_id = client.client_id
        pos, total = self._queue.get_queue_position(client_id) or (0, 0)
        wait_sec = self._queue.get_estimated_wait_seconds(client_id)
        md_content = _queue_modal_markdown(pos, total, wait_sec)

        modal = client.gui.add_modal(
            "Kimodo Demo — Please Wait",
            size="xl",
            show_close_button=False,
        )
        with modal:
            md_handle = client.gui.add_markdown(md_content)
        with self._lock:
            self._queue_modal_handles[client_id] = (modal, md_handle)

    def _show_quick_start_modal(self, client: viser.ClientHandle) -> None:
        """Show the quick start instructions modal (same as non-HF mode)."""
        with client.gui.add_modal(
            "Welcome — Quick Start",
            size="xl",
            show_close_button=True,
            save_choice="kimodo.demo.quick_start_ack",
        ) as quick_start_modal:
            client.gui.add_markdown(DEMO_UI_QUICK_START_MODAL_MD)
            client.gui.add_button("Got it (don't remind me again)").on_click(lambda _: quick_start_modal.close())

    def _show_welcome_modal(self, client: viser.ClientHandle) -> None:
        client_id = client.client_id

        def _on_start_demo(_: Any) -> None:
            modal.close()
            self._show_quick_start_modal(client)

        modal = client.gui.add_modal(
            "Welcome to Kimodo Demo",
            size="xl",
            show_close_button=True,
        )
        with modal:
            client.gui.add_markdown(_welcome_modal_markdown())
            client.gui.add_button("Start Demo").on_click(_on_start_demo)
        with self._lock:
            self._welcome_modal_handles[client_id] = modal

    def _update_all_queue_modals(self) -> None:
        with self._lock:
            handles = list(self._queue_modal_handles.items())
        for client_id, (modal, md_handle) in handles:
            pos_total = self._queue.get_queue_position(client_id)
            if pos_total is None:
                continue
            pos, total = pos_total
            wait_sec = self._queue.get_estimated_wait_seconds(client_id)
            try:
                md_handle.content = _queue_modal_markdown(pos, total, wait_sec)
            except Exception:
                pass

    def _promote_next_user(self) -> None:
        promoted_id = self._queue.promote_next()
        if promoted_id is None:
            return
        clients = self._server.get_clients()
        client = clients.get(promoted_id)
        if client is None:
            return
        with self._lock:
            old = self._queue_modal_handles.pop(promoted_id, None)
        if old is not None:
            try:
                old[0].close()
            except Exception:
                pass
        try:
            self._setup_demo_for_client(client)
        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"CUDA error while setting up client {promoted_id}: {e}")
                return
            raise
        self._start_session_timer(promoted_id)
        self._show_welcome_modal(client)
        self._update_all_queue_modals()

    def _start_session_timer(self, client_id: int) -> None:
        def on_expiry() -> None:
            self._on_session_expired(client_id)

        t = threading.Timer(self._max_seconds, on_expiry)
        t.daemon = True
        with self._lock:
            self._expiry_timers[client_id] = t
        t.start()

    def _on_session_expired(self, client_id: int) -> None:
        with self._lock:
            self._expiry_timers.pop(client_id, None)
        if not self._queue.is_active(client_id):
            return
        self._queue.remove(client_id)
        clients = self._server.get_clients()
        client = clients.get(client_id)
        if client is not None:
            try:
                with client.gui.add_modal(
                    "Session Expired",
                    size="lg",
                    show_close_button=False,
                ) as modal_ctx:
                    client.gui.add_markdown(_expiry_modal_markdown())
            except Exception:
                pass
        self._cleanup_session(client_id)
        self._promote_next_user()
