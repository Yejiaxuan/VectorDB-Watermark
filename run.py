import subprocess
import os
import time
import sys

PROJECT_ROOT = os.path.dirname(__file__)
NPM_CMD = "npm.cmd" if os.name == "nt" else "npm"


def start_backend():
    print("启动后端...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", "backend.app:app",
        "--reload", "--port", "8000"
    ])


def start_frontend():
    print("启动前端...")
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")

    # 检查并安装依赖
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("安装依赖...")
        subprocess.check_call([NPM_CMD, "install"], cwd=frontend_dir)

    return subprocess.Popen([NPM_CMD, "run", "dev"], cwd=frontend_dir)


if __name__ == "__main__":
    backend = start_backend()
    time.sleep(2)
    frontend = start_frontend()

    try:
        print("\n后端: http://localhost:8000")
        print("前端: http://localhost:5173")
        print("按 Ctrl+C 停止\n")

        backend.wait()
        if frontend:
            frontend.wait()
    except KeyboardInterrupt:
        print("\n正在关闭...")
        backend.terminate()
        if frontend:
            frontend.terminate()
        print("已退出")
