"""
快速清理8000端口的僵尸进程
用法：python scripts/kill_port.py [端口号]
"""

import sys
import psutil

def kill_port(port=8000):
    """杀死指定端口的所有进程"""
    killed = []
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    process = psutil.Process(conn.pid)
                    process_info = f"PID {conn.pid} ({process.name()})"
                    process.kill()
                    killed.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"⚠️  无法杀死进程 {conn.pid}: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return []

    if killed:
        print(f"✓ 清理了 {len(killed)} 个进程:")
        for proc in killed:
            print(f"  - {proc}")
    else:
        print(f"✓ 端口 {port} 无进程占用")

    return killed

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"🔍 检查端口 {port}...")
    kill_port(port)
