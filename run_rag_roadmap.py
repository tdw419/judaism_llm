#!/usr/bin/env python3
"""
Autonomous RAG Roadmap Executor
Loops through all 7 phases of RAG_ROADMAP.md with error handling and progress tracking
"""

import subprocess
import json
import time
from pathlib import Path
import sys

# Configuration
ROADMAP_FILE = "RAG_ROADMAP.md"
PROGRESS_FILE = "rag_progress.json"
MAX_RETRIES = 3
PHASE_TIMEOUT = 3600  # 1 hour per phase

class RAGExecutor:
    def __init__(self):
        self.progress = self.load_progress()
        self.phases = [
            {"id": "Phase 1", "name": "Data Preparation", "script": "run_phase1.py"},
            {"id": "Phase 2", "name": "Vector Database", "script": "run_phase2.py"},
            {"id": "Phase 3", "name": "RAG Query Engine", "script": "run_phase3.py"},
            {"id": "Phase 4", "name": "Interactive CLI", "script": "run_phase4.py"},
            {"id": "Phase 5", "name": "Web Interface", "script": "run_phase5.py"},
            {"id": "Phase 6", "name": "Evaluation", "script": "run_phase6_improved.py"},
            {"id": "Phase 7", "name": "Deployment", "script": "run_phase7.py"}
        ]

    def load_progress(self):
        """Load progress from file."""
        if Path(PROGRESS_FILE).exists():
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        return {"current_phase": 0, "completed_phases": [], "start_time": time.time()}

    def save_progress(self):
        """Save progress to file."""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def run_command(self, command, timeout=PHASE_TIMEOUT):
        """Run command with timeout and error handling."""
        try:
            result = subprocess.run(
                command.split(),
                timeout=timeout,
                capture_output=True,
                text=True
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def run_phase(self, phase_info):
        """Execute a single phase."""
        phase_id = phase_info["id"]
        script = phase_info["script"]

        print(f"\n{'='*60}")
        print(f"Starting {phase_id}: {phase_info['name']}")
        print(f"Script: {script}")
        print(f"{'='*60}\n")

        retries = 0
        while retries < MAX_RETRIES:
            print(f"Attempt {retries + 1}/{MAX_RETRIES}...")

            success, stdout, stderr = self.run_command(f"python3 {script}")

            if success:
                print(f"✓ {phase_id} completed successfully")
                print(f"Output: {stdout[:200]}...")
                self.progress["completed_phases"].append(phase_id)
                self.progress["current_phase"] += 1
                self.save_progress()
                return True
            else:
                print(f"✗ {phase_id} failed")
                print(f"Error: {stderr[:500]}...")
                retries += 1

                if retries < MAX_RETRIES:
                    wait_time = retries * 60
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        print(f"✗ {phase_id} failed after {MAX_RETRIES} attempts")
        return False

    def run_all(self):
        """Execute all phases."""
        print("="*60)
        print("RAG Roadmap Autonomous Executor")
        print(f"Total Phases: {len(self.phases)}")
        print(f"Starting from Phase: {self.progress['current_phase'] + 1}")
        print("="*60)

        start_phase = self.progress["current_phase"]

        for i in range(start_phase, len(self.phases)):
            phase = self.phases[i]

            if not self.run_phase(phase):
                print(f"\n✗ Roadmap execution failed at {phase['id']}")
                return False

        elapsed = time.time() - self.progress["start_time"]
        hours = elapsed / 3600

        print(f"\n{'='*60}")
        print(f"✓ All {len(self.phases)} phases completed!")
        print(f"Total time: {hours:.2f} hours")
        print(f"Progress saved to: {PROGRESS_FILE}")
        print(f"{'='*60}")

        return True

    def status(self):
        """Show current progress."""
        print(f"\nRAG Roadmap Status:")
        print(f"Current Phase: {self.progress['current_phase'] + 1}/{len(self.phases)}")
        print(f"Completed Phases: {len(self.progress['completed_phases'])}")
        print(f"Elapsed: {(time.time() - self.progress['start_time']) / 3600:.2f} hours")

        for i, phase in enumerate(self.phases):
            status = "✓" if phase["id"] in self.progress["completed_phases"] else "○"
            print(f"  {status} {phase['id']}: {phase['name']}")

if __name__ == "__main__":
    executor = RAGExecutor()

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        executor.status()
    elif len(sys.argv) > 1 and sys.argv[1] == "reset":
        Path(PROGRESS_FILE).unlink(missing_ok=True)
        print("Progress reset. Starting from Phase 1.")
    else:
        success = executor.run_all()
        sys.exit(0 if success else 1)