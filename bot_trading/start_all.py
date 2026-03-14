# -*- coding: utf-8 -*-
"""
start_all.py
Arranca el bot completo: scheduler + dashboard Streamlit.

Uso:
    python bot_trading/start_all.py

Abre dos ventanas de consola:
  1. Scheduler (proceso persistente, ejecuta paper trading cada lunes)
  2. Dashboard Streamlit en http://localhost:8501

Para detener: cierra las dos ventanas de consola o pulsa Ctrl+C en cada una.
"""

import subprocess
import sys
import pathlib
import time
import webbrowser
import os

BASE_DIR = pathlib.Path(__file__).parent.parent   # C:\TFG
BOT_DIR  = pathlib.Path(__file__).parent          # C:\TFG\bot_trading
PYTHON   = sys.executable
PORT     = 8501


def launch_scheduler():
    """Abre el scheduler en una ventana de consola nueva."""
    cmd = [
        "cmd", "/c", "start",
        "Bot Oro - Scheduler",            # titulo de la ventana
        "cmd", "/k",
        f"{PYTHON} -X utf8 {BOT_DIR / 'scheduler.py'}"
    ]
    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), shell=False)
    print(f"[OK] Scheduler iniciado (PID {proc.pid})")
    return proc


def launch_dashboard():
    """Abre el dashboard Streamlit en una ventana de consola nueva."""
    cmd = [
        "cmd", "/c", "start",
        "Bot Oro - Dashboard",
        "cmd", "/k",
        (f"{PYTHON} -m streamlit run "
         f"{BOT_DIR / 'dashboard.py'} "
         f"--server.port {PORT} "
         f"--server.headless false")
    ]
    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), shell=False)
    print(f"[OK] Dashboard iniciado (PID {proc.pid})")
    return proc


def main():
    print("=" * 55)
    print("  BOT SEMANAL DE ORO — Arrancando todo el sistema")
    print("=" * 55)

    # 1. Lanzar scheduler
    print("\n[1/2] Iniciando scheduler (paper trading automatico)...")
    launch_scheduler()
    time.sleep(2)

    # 2. Lanzar dashboard
    print("[2/2] Iniciando dashboard Streamlit...")
    launch_dashboard()
    time.sleep(6)

    # 3. Abrir navegador
    url = f"http://localhost:{PORT}"
    print(f"\n[OK] Abriendo dashboard en {url}")
    webbrowser.open(url)

    print("\n" + "=" * 55)
    print("  Sistema activo:")
    print(f"  - Scheduler    : ejecuta paper trading cada lunes 09:00")
    print(f"  - Dashboard    : {url}")
    print(f"  - Log          : bot_trading/output/scheduler.log")
    print(f"  - Paper trades : bot_trading/output/paper_trades.csv")
    print("=" * 55)
    print("\nCierra las ventanas del scheduler y dashboard para detener.")
    print("Pulsa Enter para cerrar esta ventana...")
    input()


if __name__ == "__main__":
    main()
