#!/usr/bin/env python3
"""
Roadmap Builder Daemon - Judaism LLM Edition

Autonomous daemon that executes roadmap tasks one at a time.
Each task is built, verified, and committed if acceptance passes.
"""

import json
import subprocess
import time
import os
import sys
from pathlib import Path


class RoadmapBuilderDaemon:
    """Execute roadmap tasks with automatic verification and checkpointing."""

    def __init__(self, roadmap_path=None):
        if roadmap_path is None:
            roadmap_path = Path(__file__).parent / "roadmap.json"
        self.roadmap_path = Path(roadmap_path)
        self.running = True
        self.iteration = 0
        self.max_iterations = 100
        self.retries = {}
        self.state = {}

    def load_roadmap(self):
        """Load roadmap configuration."""
        with open(self.roadmap_path, "r") as f:
            return json.load(f)

    def save_roadmap(self, roadmap):
        """Save roadmap state."""
        with open(self.roadmap_path, "w") as f:
            json.dump(roadmap, f, indent=2)

    def load_state(self):
        """Load daemon state from disk."""
        state_path = self.roadmap_path.parent / "roadmap_builder_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                return json.load(f)
        return {"iteration": 0, "retries": {}}

    def save_state(self, state):
        """Save daemon state to disk."""
        state_path = self.roadmap_path.parent / "roadmap_builder_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def run_command(self, command, cwd):
        """Run shell command and return result."""
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per command
        )
        return result

    def check_acceptance(self, task, target_repo):
        """Check if task acceptance criteria is met."""
        acceptance = task.get("acceptance", "")
        if not acceptance:
            return True

        result = self.run_command(acceptance, target_repo)
        return result.returncode == 0

    def run_build(self, task, target_repo):
        """Run task build command."""
        build_cmd = task.get("build", "")
        if not build_cmd:
            return True

        result = self.run_command(build_cmd, target_repo)
        return result.returncode == 0

    def commit_task(self, task_id, task_title, target_repo, external=False):
        """Commit completed task to git."""
        message = f"roadmap_builder: {task_id} {task_title}"
        if external:
            message += " (external)"

        result = self.run_command(
            f"git add -A && git commit -m '{message}'",
            target_repo
        )
        return result.returncode == 0

    def get_next_task(self, roadmap):
        """Find the next ready task."""
        tasks = roadmap.get("tasks", [])
        done_tasks = {t["id"] for t in tasks if t.get("status") == "done"}

        for task in tasks:
            if task.get("status") == "pending":
                deps = task.get("deps", [])
                if all(dep_id in done_tasks for dep_id in deps):
                    return task
        return None

    def ensure_branch(self, roadmap):
        """Ensure we're on the correct branch."""
        target_repo = roadmap.get("target_repo")
        branch = roadmap.get("branch", "master")

        # Check if target repo exists
        if not os.path.exists(target_repo):
            print(f"Error: target_repo {target_repo} does not exist")
            return False

        # Initialize git if needed
        git_dir = os.path.join(target_repo, ".git")
        if not os.path.exists(git_dir):
            result = self.run_command("git init && git add -A && git commit -m 'initial'", target_repo)
            if result.returncode != 0:
                print(f"Failed to initialize git in {target_repo}")
                return False

        # Check current branch and switch if needed
        result = self.run_command("git branch --show-current", target_repo)
        current_branch = result.stdout.strip()

        if current_branch != branch:
            result = self.run_command(f"git checkout -B {branch}", target_repo)
            if result.returncode != 0:
                print(f"Failed to checkout branch {branch}")
                return False

        return True

    def execute_iteration(self):
        """Execute one iteration of the roadmap."""
        self.iteration += 1
        print(f"\n=== Iteration {self.iteration} ===")

        # Load roadmap
        roadmap = self.load_roadmap()
        target_repo = roadmap.get("target_repo")
        branch = roadmap.get("branch", "master")
        max_retries = roadmap.get("max_retries", 3)

        print(f"Target: {target_repo}")
        print(f"Branch: {branch}")

        # Ensure correct branch
        if not self.ensure_branch(roadmap):
            print("Failed to ensure branch")
            return {"status": "error", "error": "branch_check_failed"}

        # Find next task
        task = self.get_next_task(roadmap)

        if task is None:
            # Check if there are pending tasks
            pending_tasks = [t for t in roadmap["tasks"] if t.get("status") == "pending"]
            if not pending_tasks:
                print("All tasks complete!")
                self.running = False
                return {"status": "complete"}

            # All pending tasks have unmet deps
            print("No ready tasks (waiting for dependencies)")
            return {"status": "waiting"}

        task_id = task["id"]
        task_title = task["title"]
        print(f"\nTask: {task_id}")
        print(f"Title: {task_title}")

        # Check if acceptance already passes (external fix)
        if self.check_acceptance(task, target_repo):
            print(f"Acceptance already met (external fix)")
            # Mark as done
            for t in roadmap["tasks"]:
                if t["id"] == task_id:
                    t["status"] = "done"
                    break
            self.save_roadmap(roadmap)
            self.commit_task(task_id, task_title, target_repo, external=True)
            # Clear retry count
            if task_id in self.retries:
                del self.retries[task_id]
            self.save_state(self.state)
            return {"status": "success", "task": task_id, "external": True}

        # Check retry count
        retry_count = self.retries.get(task_id, 0)
        if retry_count >= max_retries:
            print(f"Max retries ({max_retries}) exceeded for {task_id}")
            # Mark as blocked
            for t in roadmap["tasks"]:
                if t["id"] == task_id:
                    t["status"] = "blocked"
                    break
            self.save_roadmap(roadmap)
            return {"status": "blocked", "task": task_id}

        # Run build
        print(f"Running build...")
        if not self.run_build(task, target_repo):
            print(f"Build failed for {task_id}")
            self.retries[task_id] = retry_count + 1
            self.save_state(self.state)
            return {"status": "failed", "task": task_id, "retry": retry_count + 1}

        # Check acceptance
        print(f"Checking acceptance...")
        if self.check_acceptance(task, target_repo):
            print(f"Acceptance passed for {task_id}")
            # Mark as done
            for t in roadmap["tasks"]:
                if t["id"] == task_id:
                    t["status"] = "done"
                    break
            self.save_roadmap(roadmap)
            self.commit_task(task_id, task_title, target_repo)
            # Clear retry count
            if task_id in self.retries:
                del self.retries[task_id]
            self.save_state(self.state)
            return {"status": "success", "task": task_id}
        else:
            print(f"Acceptance failed for {task_id}")
            self.retries[task_id] = retry_count + 1
            self.save_state(self.state)
            return {"status": "failed", "task": task_id, "retry": retry_count + 1}

    def run(self):
        """Main daemon loop."""
        print("Roadmap Builder Daemon starting...")
        print(f"Roadmap: {self.roadmap_path}")

        # Load state
        self.state = self.load_state()
        self.retries = self.state.get("retries", {})
        self.iteration = self.state.get("iteration", 0)

        try:
            while self.running and self.iteration < self.max_iterations:
                result = self.execute_iteration()

                # Update state
                self.state["iteration"] = self.iteration
                self.state["retries"] = self.retries
                self.save_state(self.state)

                # Delay between iterations
                if result["status"] not in ["complete", "error"]:
                    time.sleep(60)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nRoadmap Builder Daemon finished after {self.iteration} iterations")


def main():
    """Entry point."""
    daemon = RoadmapBuilderDaemon()
    daemon.run()


if __name__ == "__main__":
    main()