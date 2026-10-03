"""One switch for every LLM call in final_method: Codex models (e.g. gpt-6-sol) or Claude models via the
Claude Code CLI (`claude`). Claude is selected by the model name: opus, sonnet, haiku, or any claude-* id
(e.g. claude-opus-5-5, claude-haiku-4-5-20251001). Requires a logged-in `codex` or `claude` CLI.

    call_json(model, prompt, schema, timeout)        -> dict (structured output matching `schema`)
    run_agent(model, prompt, cwd, add_dir, out_md, log_file, env, timeout)   -> agentic coding episode
"""
import json, shutil, subprocess, tempfile
from pathlib import Path

CLAUDE_ALIASES = {"opus", "sonnet", "haiku"}


def is_claude(model):
    m = str(model).lower()
    return m in CLAUDE_ALIASES or m.startswith("claude")


def _claude_bin():
    return shutil.which("claude") or "claude"


def call_json(model, prompt, schema, timeout=1500, read_only=True):
    """Single-turn call that returns a JSON object validated against `schema`."""
    if is_claude(model):
        cmd = [_claude_bin(), "-p", "--model", model, "--output-format", "json", "--json-schema", json.dumps(schema)]
        if read_only:
            cmd += ["--tools", ""]  # pure text-in / JSON-out, no tool use
        r = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"claude exited {r.returncode}: {r.stderr[-500:]}")
        d = json.loads(r.stdout)
        if d.get("is_error"):
            raise RuntimeError(f"claude error: {d.get('result', '')[:500]}")
        out = d.get("structured_output")
        return out if out is not None else json.loads(d["result"])
    with tempfile.TemporaryDirectory() as td:
        sp, op = Path(td) / "s.json", Path(td) / "o.json"
        sp.write_text(json.dumps(schema))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", model, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=prompt, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return json.loads(op.read_text())


def run_agent(model, prompt, cwd, add_dir, out_md, log_file, env=None, timeout=7200):
    """One autonomous coding episode: the agent may read/edit files in `cwd` and `add_dir` and run commands.
    The agent's final message is written to `out_md`; the full transcript goes to `log_file`."""
    if is_claude(model):
        cmd = [_claude_bin(), "-p", "--model", model, "--permission-mode", "bypassPermissions",
               "--add-dir", str(add_dir), "--output-format", "json"]
        r = subprocess.run(cmd, input=prompt, text=True, cwd=str(cwd), env=env, capture_output=True, timeout=timeout)
        log_file.write(r.stdout + "\n" + r.stderr)
        try:
            Path(out_md).write_text(json.loads(r.stdout).get("result", ""))
        except Exception:
            Path(out_md).write_text(r.stdout[-4000:])
        return r.returncode
    r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "--add-dir", str(add_dir),
                        "-m", model, "-C", str(cwd), "-o", str(out_md), "-"],
                       input=prompt, text=True, env=env, stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode
