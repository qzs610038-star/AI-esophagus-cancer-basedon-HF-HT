"""Legacy PreToolUse hook; no code directory currently has special protection."""
import sys, json, os, re

# DIR-20260809-002: these directories are historical migration candidates.
PROTECTED = []

# Bash commands that indicate WRITE intent to a path (not just read/search/run)
BASH_WRITE_PATTERNS = [
    # Redirection / overwrite
    r'>\s*\S*({d})',
    r'>>\s*\S*({d})',
    # PowerShell write/create
    r'Set-Content\s+.*({d})',
    r'Out-File\s+.*({d})',
    r'Add-Content\s+.*({d})',
    r'New-Item\s+.*({d})',
    r'Copy-Item\s+.*({d})',
    r'Rename-Item\s+.*({d})',
    r'Clear-Content\s+.*({d})',
    r'Set-Item\s+.*({d})',
    # Remove/delete
    r'Remove-Item\s+.*({d})',
    r'rm\s+.*({d})',
    r'del\s+.*({d})',
    # Move/rename (Unix)
    r'mv\s+.*({d})',
    # Copy into protected dir (Unix)
    r'cp\s+.*\s+.*({d})',
    r'copy\s+.*\s+.*({d})',
]

def path_hits_protected(target_path):
    """Normalize and check if path targets a protected directory."""
    normalized = os.path.normpath(target_path).replace("\\", "/").lower()
    for d in PROTECTED:
        if f"/{d}/" in normalized or normalized.startswith(f"{d}/") or normalized == d:
            return d
    return None

try:
    data = json.load(sys.stdin)
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    if tool_name in ("Edit", "Write", "MultiEdit"):
        target_path = tool_input.get("file_path", "")
        hit = path_hits_protected(target_path)
        if hit:
            print(f"BLOCKED: '{hit}/' 是受保护目录 (AGENTS.md 固定安全边界)", file=sys.stderr)
            print(f"  被拦截操作: {tool_name} -> {target_path}", file=sys.stderr)
            sys.exit(2)

    elif tool_name == "Bash":
        command = tool_input.get("command", "")
        # Only block Bash commands with write intent to protected dirs
        for d in PROTECTED:
            for pattern in BASH_WRITE_PATTERNS:
                if re.search(pattern.format(d=d), command, re.IGNORECASE):
                    print(f"BLOCKED: Bash 写操作涉及受保护目录 '{d}/' (AGENTS.md 固定安全边界)", file=sys.stderr)
                    print(f"  被拦截命令: {command}", file=sys.stderr)
                    sys.exit(2)

except Exception as e:
    print(f"hook: warn-protected-dir error (allowing): {e}", file=sys.stderr)

sys.exit(0)
