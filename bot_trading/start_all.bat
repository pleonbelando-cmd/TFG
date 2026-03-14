@echo off
:: start_all.bat
:: Arranca el bot de oro: scheduler + dashboard.
:: Doble clic para iniciar, o registrar en Task Scheduler para arranque automatico.

title Bot Semanal de Oro

cd /d "C:\TFG"

echo ============================================
echo   BOT SEMANAL DE ORO - Iniciando sistema
echo ============================================

:: Scheduler en ventana separada (se queda corriendo)
start "Bot Oro - Scheduler" cmd /k python -X utf8 bot_trading\scheduler.py

:: Esperar 3 segundos
timeout /t 3 /nobreak >nul

:: Dashboard Streamlit en ventana separada
start "Bot Oro - Dashboard" cmd /k python -m streamlit run bot_trading\dashboard.py --server.port 8501

:: Esperar a que arranque Streamlit
timeout /t 8 /nobreak >nul

:: Abrir navegador
start http://localhost:8501

echo.
echo [OK] Sistema iniciado.
echo   Scheduler  : ventana "Bot Oro - Scheduler"
echo   Dashboard  : http://localhost:8501
echo   Log        : bot_trading\output\scheduler.log
echo.
echo Cierra este script. Las ventanas del scheduler y dashboard
echo seguiran corriendo en segundo plano.
pause
