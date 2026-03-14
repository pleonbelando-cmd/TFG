# -*- coding: utf-8 -*-
"""
mt5_connector.py
Integración con MetaTrader 5 para ejecución de órdenes en XAUUSD.

Prerequisito:
    pip install MetaTrader5
    MT5 terminal abierto y logueado en Windows.

Flujo:
    1. mt5.initialize()           — conectar al terminal
    2. get_current_price()        — precio actual bid/ask
    3. compute_lot_size()         — tamaño de posición por % de riesgo
    4. get_open_position()        — comprobar si hay posición abierta
    5. close_position()           — cerrar posición si existe
    6. send_order()               — abrir nueva posición con SL/TP
    7. mt5.shutdown()             — desconectar

Modo demo (LIVE_TRADING=False): imprime la orden sin ejecutarla.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import numpy as np

from config import (
    SYMBOL, MAGIC_NUMBER, LIVE_TRADING,
    RISK_PCT_PER_TRADE, ATR_PERIOD,
    SL_ATR_MULT, TP_ATR_MULT,
)

# Importación condicional de MetaTrader5
try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:
    mt5 = None
    _HAS_MT5 = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_mt5() -> bool:
    if not _HAS_MT5:
        print("AVISO: MetaTrader5 no instalado. Ejecuta: pip install MetaTrader5")
        return False
    return True


def connect() -> bool:
    """Inicializa y conecta al terminal MT5. Devuelve True si OK."""
    if not _check_mt5():
        return False
    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() falló — {mt5.last_error()}")
        return False
    info = mt5.terminal_info()
    print(f"MT5 conectado: {info.name}  build={info.build}  servidor={mt5.account_info().server}")
    return True


def disconnect() -> None:
    if _HAS_MT5:
        mt5.shutdown()
        print("MT5 desconectado.")


def get_current_price(symbol: str = SYMBOL) -> dict | None:
    """Devuelve {'bid': ..., 'ask': ..., 'last': ...} o None si error."""
    if not _check_mt5():
        return None
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"ERROR: no se pudo obtener tick para {symbol}")
        return None
    return {"bid": tick.bid, "ask": tick.ask, "last": tick.last}


def get_open_position(symbol: str = SYMBOL, magic: int = MAGIC_NUMBER) -> object | None:
    """Devuelve la posición abierta del bot (o None si no hay)."""
    if not _check_mt5():
        return None
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == magic:
            return pos
    return None


# ─── ATR semanal ──────────────────────────────────────────────────────────────

def _compute_atr_weekly(symbol: str = SYMBOL, period: int = ATR_PERIOD) -> float:
    """
    Calcula el ATR(period) en barras semanales del símbolo MT5.
    Usa TIMEFRAME_W1.
    """
    if not _check_mt5():
        return 20.0   # fallback razonable para XAUUSD

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_W1, 0, period + 5)
    if rates is None or len(rates) < period:
        return 20.0

    import pandas as pd
    df = pd.DataFrame(rates)
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["prev_close"]),
            abs(df["low"]  - df["prev_close"]),
        )
    )
    return float(df["tr"].tail(period).mean())


# ─── Tamaño de posición ───────────────────────────────────────────────────────

def compute_lot_size(
    sl_distance: float,
    risk_pct: float = RISK_PCT_PER_TRADE,
    symbol: str = SYMBOL,
) -> float:
    """
    Calcula el lot size basado en riesgo fijo (% del capital).

    lot_size = (capital × risk_pct) / (sl_distance × valor_pip)

    Para XAUUSD: 1 lot = 100 oz; pip = 0.01 USD → valor_pip = 1 USD/lot
    Aproximación conservadora: 1 punto = 1 USD por lote.
    """
    if not _check_mt5():
        return 0.01

    account = mt5.account_info()
    if account is None:
        return 0.01

    capital   = account.equity
    sym_info  = mt5.symbol_info(symbol)
    if sym_info is None:
        return 0.01

    # Valor de un punto por lote (contract_size × point)
    point_value = sym_info.trade_contract_size * sym_info.point

    if sl_distance <= 0 or point_value <= 0:
        return 0.01

    lots = (capital * risk_pct) / (sl_distance / sym_info.point * point_value)
    lots = round(lots, 2)

    # Respetar límites del broker
    lots = max(sym_info.volume_min, min(lots, sym_info.volume_max))
    return lots


# ─── Operaciones ──────────────────────────────────────────────────────────────

def close_position(position) -> bool:
    """Cierra una posición abierta a precio de mercado."""
    if not _check_mt5():
        return False

    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return False

    # Dirección opuesta a la posición abierta
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price      = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    position.symbol,
        "volume":    position.volume,
        "type":      order_type,
        "position":  position.ticket,
        "price":     price,
        "deviation": 20,
        "magic":     MAGIC_NUMBER,
        "comment":   "bot_trading close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if not LIVE_TRADING:
        print(f"  [DEMO] CERRAR posición #{position.ticket}  {position.symbol}  "
              f"vol={position.volume}  precio={price:.2f}")
        return True

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  ERROR cerrando posición: {result.retcode} — {result.comment}")
        return False
    print(f"  Posición #{position.ticket} cerrada. Retcode={result.retcode}")
    return True


def send_order(
    signal: int,
    symbol: str = SYMBOL,
) -> bool:
    """
    Envía una orden de mercado según la señal.

    signal: +1 → BUY, -1 → SELL (no usado en esta estrategia, siempre CASH)
    SL y TP calculados como múltiplos del ATR semanal.
    """
    if signal not in (1, -1):
        print(f"  Señal {signal} -> no se abre posición.")
        return False

    if not _check_mt5():
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    atr = _compute_atr_weekly(symbol)
    sl_dist = SL_ATR_MULT * atr
    tp_dist = TP_ATR_MULT * atr

    if signal == 1:   # BUY
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
        sl         = price - sl_dist
        tp         = price + tp_dist
    else:             # SELL
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
        sl         = price + sl_dist
        tp         = price - tp_dist

    lots = compute_lot_size(sl_dist)

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lots,
        "type":      order_type,
        "price":     price,
        "sl":        round(sl, 2),
        "tp":        round(tp, 2),
        "deviation": 20,
        "magic":     MAGIC_NUMBER,
        "comment":   f"bot_trading {datetime.now():%Y%m%d}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    side_str = "BUY" if signal == 1 else "SELL"
    print(f"  Orden: {side_str}  {symbol}  lots={lots:.2f}  "
          f"precio={price:.2f}  SL={sl:.2f}  TP={tp:.2f}  ATR={atr:.2f}")

    if not LIVE_TRADING:
        print(f"  [DEMO] Orden impresa pero NO enviada (LIVE_TRADING=False)")
        return True

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  ERROR enviando orden: {result.retcode} — {result.comment}")
        return False
    print(f"  Orden ejecutada. Ticket #{result.order}  Retcode={result.retcode}")
    return True


# ─── Flujo completo ───────────────────────────────────────────────────────────

def execute_signal(signal: int, symbol: str = SYMBOL) -> None:
    """
    Ejecuta el ciclo completo:
    1. Conectar al MT5
    2. Obtener posición abierta
    3. Cerrar si la señal ha cambiado
    4. Abrir nueva posición si señal = BUY
    5. Desconectar
    """
    if not connect():
        print("No se pudo conectar a MT5.")
        return

    try:
        open_pos = get_open_position(symbol)

        # Determinar si hay que cerrar
        should_close = False
        if open_pos is not None:
            pos_side = 1 if open_pos.type == mt5.ORDER_TYPE_BUY else -1
            if pos_side != signal:
                should_close = True

        if should_close and open_pos is not None:
            print(f"  Cerrando posición abierta (señal cambió a {signal})...")
            close_position(open_pos)

        # Abrir nueva posición solo si señal es BUY (estrategia conservadora L/C)
        if signal == 1:
            if open_pos is None or should_close:
                send_order(signal, symbol)
            else:
                print(f"  Posición BUY ya abierta — sin cambios.")
        else:
            print(f"  Señal={signal} -> CASH (sin posición abierta).")

    finally:
        disconnect()


if __name__ == "__main__":
    # Test de conectividad (no ejecuta órdenes)
    if connect():
        price = get_current_price()
        print(f"Precio actual {SYMBOL}: {price}")
        disconnect()
    else:
        print("MT5 no disponible. Comprueba que el terminal está abierto.")
