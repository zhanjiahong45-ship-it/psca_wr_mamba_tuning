# Project workflow

Codex runs locally on Windows.

This project uses the global `deploy-visible-ssh-runner` skill.

## Architecture

- Global skill contains only reusable logic.
- Project-specific remote path is stored only in `.codex/remote_config.ps1`.
- Codex edits local project files.
- Codex deploys local files to the remote server.
- Codex must not directly modify remote source files.
- The remote server is only a runtime copy.

## Remote execution rules

- Every remote training job must run inside a detached tmux session.
- Never run long training directly in a foreground SSH command.
- Local terminal windows are only viewers.
- Closing local terminal windows must not stop training.
- Shutting down the local computer must not stop training.
- stdout/stderr must be saved to `$LogRoot/<run_name>.log`.
- The visible local terminal should attach to tmux by default:
  `ssh -t <RemoteHost> "tmux attach -t <Session>"`
- The tmux pane should show live output.
- The same output should be saved to log using `tee -a`.

## Debugging rules

- Codex should inspect remote logs itself.
- Do not ask the user to paste tracebacks.
- Do not use `cat <huge_log>` on full training logs.
- Logs may have tens of thousands or hundreds of thousands of lines.
- Use `analyze_remote_log.ps1`, `grep_remote.ps1`, and `tail_remote.ps1` to extract only relevant pieces.
- For full interactive review, use `watch_remote.ps1` to attach tmux or fall back to `less +F`.

## Deployment rules

- Before a formal remote run, deploy local code by default.
- Skip deploy only if the user explicitly says NoDeploy.
- Do not upload logs, outputs, checkpoints, wandb, model weights, or cache folders.
