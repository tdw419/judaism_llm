# Roadmap Builder for Judaism LLM

Autonomous execution system that works through the ROADMAP.md tasks one by one, automatically verifying and checkpointing each step.

## How It Works

The `roadmap_builder.py` daemon:
1. Reads `roadmap.json` for the task list
2. Picks the next ready task (all dependencies done)
3. Runs the `build` command
4. Verifies with the `acceptance` command
5. If accepted: marks task "done", commits to git
6. If failed: retries up to 3 times, then marks "blocked"
7. Repeats until all tasks complete

## Quick Start

### Test Run (Manual)

```bash
cd /home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder

# Run daemon manually (will exit when stuck or complete)
python3 roadmap_builder.py
```

### Production Run (Cron)

```bash
# Schedule supervisor to run every 5 minutes
hermes cron create \
  --schedule="*/5 * * * *" \
  --name="judaism_llm_roadmap" \
  --script="/home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder/roadmap_builder_supervisor.py" \
  --deliver="origin"
```

The supervisor will:
- Check if daemon is healthy every 5 minutes
- Restart it if crashed
- Alert if it's stuck (no progress after 50 iterations)

## Roadmap Structure

Tasks are defined in `roadmap.json`:

```json
{
  "target_repo": "/path/to/judaism_llm",
  "branch": "roadmap-execution",
  "max_retries": 3,
  "tasks": [
    {
      "id": "PHASE1-DOWNLOAD",
      "title": "Phase 1: Download Sefaria texts",
      "build": "python3 download_sefaria.py",
      "acceptance": "test -d sefaria_texts && [ $(find sefaria_texts -name '*.json' | wc -l) -gt 1000 ]",
      "status": "pending",
      "deps": []
    }
  ]
}
```

### Task Fields

- `id`: Unique identifier (used in deps)
- `title`: Human-readable description
- `build`: Shell command to execute (creates the change)
- `acceptance`: Shell command that exits 0 when task is done
- `status`: `pending`, `done`, or `blocked`
- `deps`: List of task IDs that must complete first

### Important Rules

1. **`acceptance` is the source of truth** - Use real checks, not placeholders
2. **`build` must be idempotent** - Re-running shouldn't break things
3. **Order via `deps`, not array order** - The daemon resolves dependencies
4. **Use absolute paths or relative to `target_repo`** - Commands run in that directory

## Current Roadmap Status

Tasks are organized by phase from ROADMAP.md:

### Phase 1: Data Acquisition
- `PHASE1-DOWNLOAD`: Download Sefaria texts
- `PHASE1-PREPARE`: Prepare JSONL training data
- `PHASE1-AUDIT`: Verify file counts and content
- `PHASE1-COMPLETE`: Mark phase complete

### Phase 2: Base Model Selection
- `PHASE2-BASEMODEL`: Document model decision

### Phase 3: Training Infrastructure
- `PHASE3-DEPS`: Document dependencies
- `PHASE3-SCRIPT`: Create unsloth_finetune.py
- `PHASE3-COMPLETE`: Mark infrastructure ready

### Phase 4: Training Execution
- `PHASE4-TRAIN`: Run fine-tuning
- `PHASE4-COMPLETE`: Save model

### Phase 5: Evaluation
- `PHASE5-EVAL`: Create test suite
- `PHASE5-RUN`: Run benchmarks
- `PHASE5-COMPLETE`: Document results

### Phase 6: Deployment
- `PHASE6-GGUF`: Export to GGUF format
- `PHASE6-OLLAMA`: Create Ollama modelfile
- `PHASE6-CLI`: Build chat interface
- `PHASE6-COMPLETE`: Mark deployed

### Final
- `PROJECT-COMPLETE`: Judaism LLM ready

## Monitoring

### Check Progress

```bash
# View current roadmap status
cat /home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder/roadmap.json | grep -E '"(id|status)"' | paste - -

# View daemon state
cat /home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder/roadmap_builder_state.json

# Check git commits on roadmap-execution branch
cd /home/jericho/projects/zion/projects/judaism_llm/judaism_llm
git log --oneline roadmap-execution
```

### Restart Daemon Manually

```bash
# Kill existing daemon
if [ -f roadmap_builder/roadmap_builder.pid ]; then
  kill $(cat roadmap_builder/roadmap_builder.pid)
fi

# Start fresh
cd /home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder
python3 roadmap_builder.py
```

## Unblocking Tasks

If a task gets blocked (max retries exceeded):

1. Investigate why it failed
2. Fix the underlying issue manually or update the task
3. Edit `roadmap.json`, change status back to `"pending"`
4. Clear retry count: delete task ID from `roadmap_builder_state.json`
5. Restart daemon

## Files

- `roadmap.json` - Task definitions and status (edited by daemon)
- `roadmap_builder.py` - Main daemon loop
- `roadmap_builder_supervisor.py` - Cron health monitor
- `roadmap_builder_state.json` - Runtime state and retry counts
- `roadmap_builder.pid` - Daemon process ID

## Guarantees

- **Resumable**: State saved in roadmap.json and state.json
- **Isolated**: All work on `roadmap-execution` branch
- **Checkpointed**: One commit per verified task
- **Idempotent**: Tasks can be re-run safely
- **Automated**: Cron keeps it alive

## Troubleshooting

### Daemon not running

```bash
# Check pid
cat roadmap_builder/roadmap_builder.pid

# Check if process exists
ps aux | grep roadmap_builder

# View logs
tail -f roadmap_builder/daemon.log  # if you configure logging
```

### Task stuck

Check `roadmap_builder_state.json`:
- High `iteration` count (>50) indicates stall
- High `retries` for a task means repeated failures

### Git conflicts

The daemon works on `roadmap-execution` branch. If you're working on `master`:
```bash
git checkout roadmap-execution
# Review changes
git checkout master
git merge roadmap-execution  # when ready
```

## Integration with Hermes

The supervisor is designed to run via Hermes cron:

```bash
hermes cron create \
  --schedule="*/5 * * * *" \
  --name="judaism_llm_roadmap" \
  --script="/home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder/roadmap_builder_supervisor.py" \
  --deliver="origin"
```

The `--deliver="origin"` flag sends status updates to your current chat (if connected via Gateway).

## License

MIT (same as parent project)