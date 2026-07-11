#!/usr/bin/env python3
"""
PFMval 远程命令服务（服务器端，历史实现，已禁用）
=======================
安全的 HTTP 命令执行服务，带 Token 鉴权 + 命令白名单 + IP 过滤。

用法:
  python deploy/cmd_server.py --port 8080 --token my-secret-token
  python deploy/cmd_server.py --port 8080 --token my-secret-token --allowed-ip 1.2.3.4

环境变量:
  CMD_SERVER_PORT=8080
  CMD_SERVER_TOKEN=my-secret-token
  CMD_SERVER_ALLOWED_IP=1.2.3.4
"""

import argparse
import base64
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_PROJECT_ROOT))

# ── 安全配置 ──────────────────────────────────────────────
ALLOWED_COMMANDS = [
    # 训练脚本（Python 313）
    "train_histogene_uni_tokens.py",
    "train_histogene_uni_tokens_augmix.py",
    "train_histogene_uni_tokens_augmix_tv.py",
    "train_histogene_virchow2_tokens.py",
    "train_histogene_virchow2_tokens_augmix.py",
    "train_histogene_virchow2_tokens_augmix_tv.py",
    "egnv2_uni_tokens.py",
    # 部署脚本
    "deploy/push.sh",
    "deploy/pull.sh",
    # 分析/报告
    "generate_full_report.py",
    # EGN-v2 训练（PyG 环境 → 不同 Python）
    # 通过 run_egnv2_training.bat 间接启动
]

ALLOWED_PREFIXES = [
    "python train_",
    "python generate_",
    "python3 train_",
    "python3 generate_",
    "bash deploy/",
    "C:/Program Files/Python313/python.exe",
    "D:/conda_envs/pfmval_py310/python.exe",
]

BLOCKED_KEYWORDS = [
    "rm -rf", "rm  -rf", "del /F", "del /S", "DEL /F", "DEL /S",
    "format ", "FORMAT ",
    "shutdown", "SHUTDOWN",
    "> /dev/", "> /etc/", "C:\\Windows", "C:\\WINDOWS",
    "curl ", "wget ", "Invoke-WebRequest", "iwr ",
]

SERVER_START_TIME = time.time()
TOKEN_HASH = None
ALLOWED_IP = None


class CommandHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self):
        """哈希比对 Token，防时序攻击"""
        token = (self.path.split("token=")[-1].split("&")[0]
                 if "token=" in self.path else "")
        return (TOKEN_HASH is not None
                and hashlib.sha256(token.encode()).digest() == TOKEN_HASH)

    def _check_ip(self):
        if ALLOWED_IP is None:
            return True
        client_ip = self.client_address[0]
        return client_ip == ALLOWED_IP

    def do_GET(self):
        if not self._check_ip():
            self._send_json(403, {"error": "IP not allowed"})
            return

        path = self.path.split("?")[0]

        if path == "/ping":
            self._send_json(200, {
                "status": "ok",
                "server": platform.node(),
                "user": os.environ.get("USERNAME", "?"),
                "cwd": os.getcwd(),
                "uptime": round(time.time() - SERVER_START_TIME),
            })
            return

        if path == "/shutdown":
            if not self._check_auth():
                self._send_json(403, {"error": "Forbidden"})
                return
            self._send_json(200, {"status": "shutting down"})
            print("[SHUTDOWN] 收到合法关闭指令，5 秒后退出...")
            sys.stdout.flush()
            # 在另一个线程中关闭，避免阻塞当前响应
            import threading
            threading.Thread(target=self._delayed_shutdown, daemon=True).start()
            return

        self._send_json(404, {"error": "GET not supported, use POST /exec"})

    def _delayed_shutdown(self):
        time.sleep(2)
        # 强制退出整个进程
        os._exit(0)

    def do_POST(self):
        if not self._check_ip():
            self._send_json(403, {"error": "IP not allowed"})
            return

        if not self._check_auth():
            self._send_json(403, {"error": "Forbidden"})
            return

        path = self.path.split("?")[0]

        if path == "/exec":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return

            cmd = payload.get("cmd", "").strip()
            if not cmd:
                self._send_json(400, {"error": "Empty command"})
                return

            # ── 安全校验 ──
            if not self._validate_command(cmd):
                self._send_json(403, {
                    "error": "Command rejected by security policy",
                    "cmd": cmd,
                })
                return

            # ── 执行 ──
            print(f"[EXEC] {cmd}")
            sys.stdout.flush()

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    timeout=7200,  # 2小时超时
                    cwd=str(_PROJECT_ROOT),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                stdout = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")
                self._send_json(200, {
                    "ok": result.returncode == 0,
                    "code": result.returncode,
                    "stdout": stdout[-50000:],  # 限制输出长度
                    "stderr": stderr[-10000:],
                })
            except subprocess.TimeoutExpired:
                self._send_json(504, {"error": "Command timeout (>2h)"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif path == "/upload":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return

            filename = payload.get("file", "").strip()
            content_b64 = payload.get("data", "").strip()
            if not filename or not content_b64:
                self._send_json(400, {"error": "Need 'file' and 'data' (base64)"})
                return

            # 路径安全：禁止 .. 穿越，限制在项目目录
            dest = (_PROJECT_ROOT / filename).resolve()
            if not str(dest).startswith(str(_PROJECT_ROOT.resolve())):
                self._send_json(403, {"error": "Path traversal blocked"})
                return

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = base64.b64decode(content_b64)
                dest.write_bytes(data)
                print(f"[UPLOAD] {filename} ({len(data)} bytes)")
                sys.stdout.flush()
                self._send_json(200, {
                    "ok": True,
                    "file": filename,
                    "size": len(data),
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Unknown endpoint"})

    def _validate_command(self, cmd):
        # 检查黑名单关键词
        cmd_lower = cmd.lower()
        for kw in BLOCKED_KEYWORDS:
            if kw.lower() in cmd_lower:
                return False

        # 检查白名单前缀
        for prefix in ALLOWED_PREFIXES:
            if cmd.startswith(prefix):
                return True

        # 检查是否包含已知的脚本文件名
        for allowed in ALLOWED_COMMANDS:
            if allowed in cmd:
                return True

        return False


def legacy_main():
    global TOKEN_HASH, ALLOWED_IP

    parser = argparse.ArgumentParser(description="PFMval 远程命令服务")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CMD_SERVER_PORT", 8080)))
    parser.add_argument("--token", default=os.environ.get("CMD_SERVER_TOKEN", ""))
    parser.add_argument("--allowed-ip", default=os.environ.get("CMD_SERVER_ALLOWED_IP", ""))
    args = parser.parse_args()

    if not args.token:
        print("ERROR: 必须提供 --token 参数或设置 CMD_SERVER_TOKEN 环境变量")
        sys.exit(1)

    TOKEN_HASH = hashlib.sha256(args.token.encode()).digest()
    if args.allowed_ip:
        ALLOWED_IP = args.allowed_ip
        print(f"[SEC] IP 白名单: {ALLOWED_IP}")

    print(f"[START] 项目目录: {_PROJECT_ROOT}")
    print(f"[START] 监听端口: {args.port}")
    print(f"[START] Token 哈希: {TOKEN_HASH.hex()[:16]}...")
    print(f"[START] 安全策略: 黑名单 {len(BLOCKED_KEYWORDS)} 项, 白名单前缀 {len(ALLOWED_PREFIXES)} 项")
    print(f"[READY] 等待命令... (GET /ping 探测, POST /exec 执行, GET /shutdown 关闭)")
    print(f"[TIP] 关闭服务: GET /shutdown?token=<your-token>")
    print("-" * 50)

    server = HTTPServer(("0.0.0.0", args.port), CommandHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] 收到 Ctrl+C，退出。")
        server.server_close()


def main():
    print("[BLOCKED] deploy/cmd_server.py is a historical HTTP direct-transport implementation.")
    print("[BLOCKED] Current project policy permits server exchange only through the configured Gitee remote.")
    print("[INFO] Use deploy/pfmval_ops.py job/result envelopes instead.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
