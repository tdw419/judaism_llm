#!/usr/bin/env python3
"""
Roadmap Builder Supervisor

Monitors and restarts the roadmap daemon if it crashes or gets stuck.
Intended to be run via cron (every 5 minutes).
"""

import json
import os
import subprocess
import sys
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta


class Supervisor:
    """Monitor and supervise the roadmap daemon."""

    def __init__(self):
        self.daemon_script = Path(__file__).parent / "roadmap_builder.py"
        self.state_path = Path(__file__).parent / "roadmap_builder_state.json"
        self.pid_path = Path(__file__).parent / "roadmap_builder.pid"
        self.roadmap_path = Path(__file__).parent / "roadmap.json"
        self.max_iterations = 100
        self.stall_threshold = 50
        self.max_age_hours = 24

    def load_state(self):
        """Load daemon state."""
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                return json.load(f)
        return {"iteration": 0}

    def load_roadmap(self):
        """Load roadmap."""
        with open(self.roadmap_path, "r") as f:
            return json.load(f)

    def check_pid_alive(self, pid):
        """Check if a process is alive."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def get_pid(self):
        """Read pid file."""
        if self.pid_path.exists():
            with open(self.pid_path, "r") as f:
                return int(f.read().strip())
        return None

    def start_daemon(self):
        """Start the daemon."""
        print("Starting daemon...")
        proc = subprocess.Popen(
            [sys.executable, str(self.daemon_script)],
            stdout=self.daemon_path / "daemon.log",
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        with open(self.pid_path, "w") as f:
            f.write(str(proc.pid))
        return proc.pid

    def check_health(self):
        """Check daemon health."""
        # Check if daemon is running
        pid = self.get_pid()
        if pid is None or not self.check_pid_alive(pid):
            print("Daemon not running - will restart")
            return "dead"

        # Load state
        state = self.load_state()
        iteration = state.get("iteration", 0)
        last_update = state.get("last_update", None)

        # Check iteration count
        if iteration > self.max_iterations:
            print(f"Max iterations ({self.max_iterations}) exceeded")
            return "stuck"

        # Check for stall
        if iteration > self.stall_threshold:
            # Check if progress has stalled
            roadmap = self.load_roadmap()
            last_working = state.get("last_working_step", 0)
            current_done = sum(1 for t in roadmap["tasks"] if t.get("status") == "done")

            if current_done == last_working and iteration > self.stall_threshold + 10:
                print(f"Stalled at iteration {iteration}, no progress")
                return "stalled"

        # Check for stale state
        if last_update:
            last_update_time = datetime.fromisoformat(last_update)
            age = datetime.now() - last_update_time
            if age > timedelta(hours=self.max_age_hours):
                print(f"State stale ({age.total_seconds()/3600:.1f} hours old)")
                return "stale"

        return "healthy"

    def supervise(self):
        """Main supervision loop."""
        health = self.check_health()

        if health != "healthy":
            print(f"Unhealthy: {health}")

            # Kill existing daemon if running
            pid = self.get_pid()
            if pid and self.check_pid_alive(pid):
                print(f"Killing daemon (pid {pid})")
                os.kill(pid, signal.SIGTERM)
                time.sleep(5)
                if self.check_pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)

            # Start new daemon
            self.start_daemon()
            print("Daemon restarted")
            return 1  # Unhealthy

        print("Daemon healthy")
        return 0  # Healthy


def main():
    """Entry point."""
    supervisor = Supervisor()
    return supervisor.supervise()


if __name__ == "__main__":
    sys.exit(main())