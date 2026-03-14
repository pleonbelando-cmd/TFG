@echo off
:: register_task_scheduler.bat
:: Registra el bot de oro en el Task Scheduler de Windows para que
:: arranque automaticamente cuando el usuario inicia sesion.
::
:: EJECUTAR COMO ADMINISTRADOR

echo Registrando Bot de Oro en Windows Task Scheduler...

:: Nombre de la tarea
set TASK_NAME=BotOroSemanal

:: Ruta al bat de arranque
set BAT_PATH=C:\TFG\bot_trading\start_all.bat

:: Borrar tarea si ya existe
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: Crear tarea: arrancar al iniciar sesion, con retraso de 60 segundos
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "%BAT_PATH%" ^
  /sc ONLOGON ^
  /delay 0001:00 ^
  /ru "%USERNAME%" ^
  /rl HIGHEST ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Tarea registrada correctamente.
    echo   Nombre : %TASK_NAME%
    echo   Accion : %BAT_PATH%
    echo   Cuando : Al iniciar sesion (con 1 min de retraso)
    echo.
    echo Para ver la tarea: Busca "Programador de tareas" en Windows
    echo Para eliminarla:   schtasks /delete /tn %TASK_NAME% /f
) else (
    echo.
    echo [ERROR] No se pudo registrar la tarea.
    echo Asegurate de ejecutar este script como Administrador.
)

pause
