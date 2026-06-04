#!/usr/bin/env python3
"""
PFMval 远程命令客户端
=======================
向服务器发送命令并获取结果。支持交互模式和单次模式。

用法:
  # 单次执行
  python deploy/cmd_client.py --host <SERVER_IP> --port 8080 --token my-token "python train_xxx.py --epochs 5"

  # 交互模式
  python deploy/cmd_client.py --host <SERVER_IP> --port 8080 --token my-token

  # 从环境变量或 secrets.sh 读取配置
  source deploy/secrets.sh  # 含 CMD_SERVER_HOST / CMD_SERVER_TOKEN
  python deploy/cmd_client.py --port 8080 "train command"
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import urllib.error

# ── 检测是否为后台训练命令 ──
BG_PATTERNS = [r"--epochs\s+\d{2,}", r"--epochs\s+[1-9]\d\d"]
LONG_RUNNING_TIMEOUT = 3600  # 对于训练命令，只等首行输出


def send_request(host, port, token, endpoint, payload=None, timeout=30):
    """发送 HTTP 请求到命令服务器"""
    url = f"http://{host}:{port}{endpoint}?token={token}"
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": f"HTTP {e.code}: {body[:200]}"}
    except urllib.error.URLError as e:
        return {"error": f"连接失败: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def is_long_running(cmd):
    for pat in BG_PATTERNS:
        if re.search(pat, cmd):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="PFMval 远程命令客户端")
    parser.add_argument("--host", default=os.environ.get("CMD_SERVER_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CMD_SERVER_PORT", 8080)))
    parser.add_argument("--token", default=os.environ.get("CMD_SERVER_TOKEN", ""))
    parser.add_argument("command", nargs="*", help="要执行的命令（留空进入交互模式）")
    args = parser.parse_args()

    token = args.token or os.environ.get("PFMVAL_SERVER_PASSWORD", "")
    if not token:
        print("ERROR: 需要 --token 或设置 CMD_SERVER_TOKEN / PFMVAL_SERVER_PASSWORD 环境变量")
        print("TIP: source deploy/secrets.sh && source deploy/config.sh")
        sys.exit(1)

    # ── Ping 检查 ──
    print(f"连接到 {args.host}:{args.port} ...")
    resp = send_request(args.host, args.port, token, "/ping")
    if "error" in resp:
        print(f"FAIL: {resp['error']}")
        sys.exit(1)
    print(f"服务器: {resp.get('server', '?')} @ {resp.get('user', '?')}")
    print(f"工作目录: {resp.get('cwd', '?')}")
    print(f"运行时间: {resp.get('uptime', 0)}s")
    print("-" * 50)

    cmd = " ".join(args.command).strip() if args.command else ""

    if cmd:
        # ── 单次模式 ──
        run_command(args.host, args.port, token, cmd)
    else:
        # ── 交互模式 ──
        print("交互模式 - 输入命令后回车执行。")
        print("  :quit / :q  退出")
        print("  :shutdown   关闭服务器")
        print("  :ping       检查服务器状态")
        print()
        while True:
            try:
                cmd = input("CMD> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                break

            if not cmd:
                continue
            if cmd in (":quit", ":q"):
                break
            if cmd == ":ping":
                resp = send_request(args.host, args.port, token, "/ping")
                print(f"  在线 {resp.get('uptime', 0)}s")
                continue
            if cmd == ":shutdown":
                resp = send_request(args.host, args.port, token, "/shutdown")
                print(f"  {resp}")
                break
            if cmd.startswith(":upload "):
                local_path = cmd[len(":upload "):].strip()
                upload_file(args.host, args.port, token, local_path)
                continue

            run_command(args.host, args.port, token, cmd)


def upload_file(host, port, token, local_path):
    """上传单个文件到服务器的项目目录"""
    local = Path(local_path)
    if not local.exists():
        print(f"[ERROR] 文件不存在: {local_path}")
        return
    data_b64 = base64.b64encode(local.read_bytes()).decode("ascii")
    print(f"上传 {local.name} ({len(local.read_bytes())} bytes) ...")
    resp = send_request(host, port, token, "/upload",
                        {"file": local_path.replace("\\", "/"), "data": data_b64},
                        timeout=60)
    if resp.get("ok"):
        print(f"  [OK] {resp.get('file')} ({resp.get('size')} bytes)")
    else:
        print(f"  [FAIL] {resp.get('error', resp)}")


def run_command(host, port, token, cmd):
    long_run = is_long_running(cmd)
    timeout = LONG_RUNNING_TIMEOUT if long_run else 120

    if long_run:
        print(f"[训练命令，超时={timeout}s，等待初始输出...]")

    t0 = time.time()
    resp = send_request(host, port, token, "/exec", {"cmd": cmd}, timeout=timeout)
    elapsed = time.time() - t0

    if resp.get("ok"):
        print(f"[OK, rc=0, {elapsed:.1f}s]")
    elif "error" in resp:
        print(f"[ERROR] {resp['error']}")
    else:
        print(f"[FAIL, rc={resp.get('code', '?')}, {elapsed:.1f}s]")

    stdout = resp.get("stdout", "")
    stderr = resp.get("stderr", "")

    if stdout:
        print(stdout)
    if stderr:
        print(f"[STDERR]\n{stderr}")

    if stdout or stderr:
        print(f"[--- 耗时 {elapsed:.1f}s ---]")


if __name__ == "__main__":
    main()
