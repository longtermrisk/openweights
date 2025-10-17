# OpenWeights CLI

This directory contains the CLI implementation for OpenWeights.

## Structure

- `__init__.py` - Main entry point that dispatches to subcommands
- `__main__.py` - Allows running as a module: `python -m openweights.cli`
- `common.py` - Shared utilities for provider abstraction, file sync, and remote operations
- `ssh.py` - Interactive SSH shell with live bidirectional file sync
- `exec.py` - Execute commands on remote GPU with file sync

## Usage

The CLI is exposed as the `ow` command after installation:

```bash
# Interactive shell with file sync
ow ssh --gpu L40 --image nielsrolf/ow-default:v0.7

# Execute a command on remote GPU
ow exec --gpu L40 --mount .:/workspace python train.py config.json

# With environment variables
ow exec --env-file .env --gpu H100 python train.py

# Multiple mounts
ow ssh --mount ~/data:/data --mount ~/code:/code
```

## Commands

### `ow ssh`

Opens an interactive shell on a remote GPU with live bidirectional file sync using Unison.

**Key features:**
- Bidirectional file sync (changes on either side are synced)
- Automatic editable install if `pyproject.toml` exists
- Interactive prompt blocks until initial sync completes
- Optional machine termination on exit

### `ow exec`

Executes a command on a remote GPU with file sync.

**Key features:**
- One-shot file sync before execution
- Automatic machine termination after execution (unless `--no-terminate`)
- Can skip file sync with `--no-sync` for faster execution
- All command arguments are passed through

## Provider Abstraction

The CLI uses a provider abstraction layer (`Provider` class in `common.py`) that allows supporting multiple cloud providers. Currently implemented:

- **RunpodProvider**: Uses RunPod for GPU instances

The abstraction makes it easy to add support for other providers (AWS, GCP, Lambda Labs, etc.) by implementing the `Provider` interface.

## File Sync

File synchronization is handled by Unison for bidirectional sync:

- **ssh command**: Continuous watch mode - changes are synced in real-time
- **exec command**: One-shot sync before execution

The sync system uses a sentinel file (`.ow_sync/busy`) to ensure the remote shell prompt blocks until the initial sync completes, guaranteeing files are up-to-date before any commands run.
