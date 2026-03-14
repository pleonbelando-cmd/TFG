# -*- coding: utf-8 -*-
"""
notifier.py
Notificaciones por email para el bot semanal de oro.

Envia un correo cada lunes cuando el bot ejecuta el ciclo de paper trading:
  - Si hay una operacion cerrada: resultado (P&L, % de acierto)
  - La nueva señal de la semana (BUY / FLAT / CASH)
  - Resumen del capital virtual acumulado

Configuracion en .env:
    EMAIL_FROM=tu_email@gmail.com
    EMAIL_PASSWORD=abcd efgh ijkl mnop   (App Password de Gmail)
    EMAIL_TO=tu_email@gmail.com
"""

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
log = logging.getLogger("bot_scheduler")

# Cargar .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
except ImportError:
    pass


def _get_config() -> dict:
    return {
        "from":     os.getenv("EMAIL_FROM", ""),
        "password": os.getenv("EMAIL_PASSWORD", ""),
        "to":       os.getenv("EMAIL_TO", ""),
        "smtp":     os.getenv("EMAIL_SMTP", "smtp.gmail.com"),
        "port":     int(os.getenv("EMAIL_PORT", "587")),
    }


def _build_html(result: dict, state: dict) -> str:
    """Construye el cuerpo HTML del correo."""

    fecha = result["current_date"]
    if hasattr(fecha, "date"):
        fecha = fecha.date()

    signal = result["new_signal"]
    prob   = result["prob_ensemble"] * 100
    precio = result["current_price"]
    capital = result["virtual_capital"]
    retorno = result["total_return_pct"]
    n_trades = result["n_trades"]

    # Colores por señal
    signal_color = {"BUY": "#27ae60", "FLAT": "#f39c12", "CASH": "#e74c3c"}.get(signal, "#7f8c8d")
    signal_emoji = {"BUY": "🟢", "FLAT": "🟡", "CASH": "🔴"}.get(signal, "⚪")

    # Bloque de operacion cerrada (si hay)
    closed_html = ""
    ct = result.get("closed_trade")
    if ct and ct.get("signal_str") == "BUY":
        pnl_pct = ct["pnl_pct"]
        pnl_usd = ct["pnl_usd"]
        exito   = pnl_pct > 0
        color_pnl = "#27ae60" if exito else "#e74c3c"
        icono     = "✅" if exito else "❌"
        resultado = "EXITOSA" if exito else "PERDIDA"
        closed_html = f"""
        <div style="background:#f8f9fa;border-left:4px solid {color_pnl};padding:12px;margin:12px 0;border-radius:4px;">
          <b>{icono} Operacion anterior cerrada — {resultado}</b><br>
          Entrada: <b>${ct['entry_price']:,.2f}</b> &nbsp;|&nbsp;
          Salida: <b>${ct['exit_price']:,.2f}</b><br>
          P&amp;L: <b style="color:{color_pnl}">{pnl_pct:+.2f}% ({pnl_usd:+.2f} USD)</b>
        </div>
        """

    # Bloque de nueva posicion abierta
    opened_html = ""
    ot = result.get("opened_trade")
    if ot:
        opened_html = f"""
        <div style="background:#eaf4fb;border-left:4px solid #2980b9;padding:12px;margin:12px 0;border-radius:4px;">
          <b>📂 Nueva posicion abierta: {ot['signal_str']}</b><br>
          Precio entrada: <b>${ot['entry_price']:,.2f}</b><br>
          Capital asignado: <b>${ot['capital']:,.2f}</b>
        </div>
        """
    elif signal == "FLAT":
        opened_html = """
        <div style="background:#fef9e7;border-left:4px solid #f39c12;padding:12px;margin:12px 0;border-radius:4px;">
          <b>🟡 Sin operacion esta semana</b><br>
          Señal FLAT — probabilidad en zona neutra. Mantener posicion actual.
        </div>
        """

    retorno_color = "#27ae60" if retorno >= 0 else "#e74c3c"

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;color:#2c3e50;">

      <div style="background:linear-gradient(135deg,#b8860b,#ffd700);padding:20px;border-radius:8px 8px 0 0;">
        <h1 style="color:white;margin:0;font-size:22px;">🥇 Bot Semanal de Oro</h1>
        <p style="color:rgba(255,255,255,0.9);margin:4px 0 0;">
          Informe semanal · {fecha} · XAUUSD
        </p>
      </div>

      <div style="background:white;padding:20px;border:1px solid #e0e0e0;border-top:none;">

        <!-- Señal principal -->
        <div style="text-align:center;padding:20px 0;">
          <div style="font-size:36px;font-weight:bold;color:{signal_color};">
            {signal_emoji} {signal}
          </div>
          <div style="color:#7f8c8d;font-size:14px;margin-top:4px;">
            Prob. Ensemble: <b>{prob:.1f}%</b> &nbsp;·&nbsp;
            Precio oro: <b>${precio:,.2f}</b>
          </div>
        </div>

        {closed_html}
        {opened_html}

        <!-- Resumen capital -->
        <div style="background:#f8f9fa;padding:16px;border-radius:6px;margin-top:12px;">
          <b>📊 Resumen del portfolio virtual</b>
          <table style="width:100%;margin-top:8px;border-collapse:collapse;">
            <tr>
              <td style="padding:4px 0;color:#7f8c8d;">Capital actual:</td>
              <td style="text-align:right;font-weight:bold;">${capital:,.2f} USD</td>
            </tr>
            <tr>
              <td style="padding:4px 0;color:#7f8c8d;">Retorno acumulado:</td>
              <td style="text-align:right;font-weight:bold;color:{retorno_color};">{retorno:+.2f}%</td>
            </tr>
            <tr>
              <td style="padding:4px 0;color:#7f8c8d;">Operaciones totales:</td>
              <td style="text-align:right;font-weight:bold;">{n_trades}</td>
            </tr>
          </table>
        </div>

        <p style="color:#95a5a6;font-size:11px;margin-top:20px;text-align:center;">
          Este correo es generado automaticamente por el bot de paper trading.<br>
          No constituye asesoramiento financiero.
        </p>
      </div>

    </body></html>
    """
    return html


def send_notification(result: dict, state: dict) -> bool:
    """
    Envia el correo de notificacion semanal.
    Devuelve True si se envio correctamente, False si hay error.
    """
    cfg = _get_config()

    if not cfg["from"] or not cfg["password"] or not cfg["to"]:
        log.warning("Email no configurado. Añade EMAIL_FROM, EMAIL_PASSWORD y EMAIL_TO al .env")
        return False

    try:
        fecha = result["current_date"]
        if hasattr(fecha, "date"):
            fecha = fecha.date()

        signal = result["new_signal"]
        signal_emoji = {"BUY": "🟢", "FLAT": "🟡", "CASH": "🔴"}.get(signal, "⚪")

        # Asunto
        ct = result.get("closed_trade")
        if ct and ct.get("signal_str") == "BUY":
            exito  = ct["pnl_pct"] > 0
            icono  = "✅" if exito else "❌"
            subject = (f"{icono} Bot Oro {fecha} — "
                       f"Cerrada {ct['pnl_pct']:+.2f}% | "
                       f"Nueva señal: {signal_emoji} {signal}")
        else:
            subject = f"{signal_emoji} Bot Oro {fecha} — Señal: {signal}"

        # Mensaje
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg["from"]
        msg["To"]      = cfg["to"]

        html_body = _build_html(result, state)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # Enviar
        with smtplib.SMTP(cfg["smtp"], cfg["port"]) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["from"], cfg["password"])
            server.sendmail(cfg["from"], cfg["to"], msg.as_bytes())

        log.info(f"Email enviado a {cfg['to']} — Asunto: {subject}")
        return True

    except Exception as e:
        log.error(f"Error enviando email: {e}")
        return False


if __name__ == "__main__":
    # Test con datos ficticios
    result_test = {
        "current_date":    datetime.now(),
        "current_price":   3050.0,
        "new_signal":      "BUY",
        "prob_ensemble":   0.61,
        "virtual_capital": 10_340.0,
        "total_return_pct": 3.4,
        "n_trades":        2,
        "closed_trade": {
            "signal_str":  "BUY",
            "entry_price": 2980.0,
            "exit_price":  3050.0,
            "pnl_pct":     2.35,
            "pnl_usd":     235.0,
        },
        "opened_trade": {
            "signal_str":  "BUY",
            "entry_price": 3050.0,
            "capital":     10_340.0,
        },
    }
    state_test = {"open": True, "signal": 1, "signal_str": "BUY"}
    ok = send_notification(result_test, state_test)
    print("Email enviado:" if ok else "Error (revisa .env)")
