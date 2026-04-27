//+------------------------------------------------------------------+
//|                                             FxGuru-V3.0.mq5      |
//|                                   Copyright 2026, FxGuru Team    |
//|                       https://trading-bot-fxwd.onrender.com      |
//+------------------------------------------------------------------+
//  VERSION HISTORY
//  ───────────────
//  v1.00  Initial release (FxGuru1Link)
//  v2.00  Multi-symbol hub, corrected data-push schema, legacy IR fallback
//  v3.00  Position-event polling, fundamental analyst filter, IRMissingLastLog
//
//  ARCHITECTURE
//  ────────────
//  Single-chart EA managing up to 8 symbols via a for-loop hub.
//
//  OnTimer  → State machine: BOOT → POLLING ⇄ DATA_PUSH
//                                       ↓ ERROR_RECOVERY → BOOT
//
//             Inside STATE_POLLING every timer tick runs in this ORDER:
//               1. PollForPositionEvents()   ← state reconciliation FIRST
//               2. PollForSignals()          ← entry signals SECOND
//             This order prevents the EA acting on a new signal for a symbol
//             whose position was just closed by the bot on the same tick.
//
//  OnTick   → ManageOpenPosition() for each open position.
//             Skips any position whose ticket is in BotClosedTickets[].
//
//  Four SMC management stages (persisted via comment across restarts):
//    Stage 1  Break-Even      : tick-accurate, one-shot, at 1.0R
//    Stage 2  Partial TP      : one-shot at 1.5R (BE must be set first)
//    Stage 3  ATR Trail       : candle-gated (new H1 bar only)
//    Stage 4  Stagnation Exit : candle-gated, at ≥Stagnation_R, N bars
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, FxGuru Team"
#property link      "https://trading-bot-fxwd.onrender.com"
#property version   "3.00"
#property description "Multi-Symbol SMC Hub v3 | Position Events | Fundamental Filter | ATR Trail"
#property strict

#include <Trade\Trade.mqh>

//--- State Machine Enum
enum ENUM_EA_STATE {
   STATE_BOOT,
   STATE_HANDSHAKE_PENDING,
   STATE_POLLING,
   STATE_DATA_PUSH,
   STATE_ERROR_RECOVERY
};

//==========================================================================
//  INPUT PARAMETERS
//==========================================================================

input group "=== API Configuration ==="
input string ApiBaseUrl               = "https://trading-bot-fxwd.onrender.com"; // Base URL (no trailing slash)
input string SymbolList               = "EURUSD,GBPUSD,USDJPY,XAUUSD";           // Comma-separated symbols (max 8)
input int    PollIntervalSeconds      = 3;     // Entry signal poll frequency (seconds)
input int    EventPollIntervalSeconds = 3;     // Position event poll frequency (seconds)
input int    DataPushMinutes          = 5;     // OHLCV push frequency (minutes)
input bool   EnableApiKeyHeader       = false; // Send X-API-KEY header on all requests
input string ApiKey                   = "";    // API key value (if EnableApiKeyHeader = true)

input group "=== Risk Management ==="
input double RiskPercent              = 1.0;   // % of account balance to risk per trade
input double FixedLotFallback         = 0.01;  // Fallback lot when SL distance is zero

input group "=== Trading Rules ==="
input long   MagicNumber              = 234000;
input int    SlippagePoints           = 20;    // Max execution slippage (points)
input double MaxSpreadPoints          = 50.0;  // Max allowed spread (points)
input bool   AllowLong                = true;  // Allow BUY signals
input bool   AllowShort               = true;  // Allow SELL signals
input bool   OnePositionPerSymbol     = true;  // Block new trade if one is already open
input bool   UseDynamicSLTP           = true;  // Use SL/TP values from API signal

input group "=== SMC Brain — Stage Parameters ==="
input double SlippagePips             = 1.5;  // Pips above/below entry for Break-Even SL
input double BE_Trigger_R             = 1.0;  // R-multiple to trigger Break-Even
input double Partial_TP_R             = 1.5;  // R-multiple to scale out 50%
input double Stagnation_R             = 2.0;  // R-multiple floor to activate stagnation timer
input int    Stagnation_Candles       = 8;    // Consecutive bars with no new extreme → close
input int    ATR_Period               = 14;   // ATR lookback (H1 candles)
input double ATR_Buffer_Mult          = 0.5;  // Trail: swing ± (mult × ATR), standard
input double ATR_Tight_Mult           = 0.25; // Trail: tighter multiplier after partial TP
input int    Swing_Lookback           = 12;   // Bars to search for structural swing high/low

input group "=== Fundamental Filter ==="
input bool   FundamentalFilterEnabled     = true;   // Enable fundamental alignment check
input string FundamentalFilterMode        = "SOFT";
// OFF  : Ignore fundamental_rating. Trade all signals normally.
// SOFT : Trade the signal but halve lot size when fundamental_rating = -1.
// HARD : Reject signal entirely when fundamental_rating = -1.
input int    FundamentalHardMinConviction = 0;
// Minimum conviction level for HARD block to apply.
// 0 = block on any -1 rating regardless of conviction.
// 1 = block only when conviction is "moderate" or "strong".
// 2 = block only when conviction is "strong".
// When HARD is set and conviction is BELOW this threshold, SOFT (halved lot)
// applies instead — prevents one weak data source from blocking a good setup.

input group "=== System ==="
input bool   DebugLogs                    = true;  // Enable verbose logging

//==========================================================================
//  GLOBAL VARIABLES
//==========================================================================

ENUM_EA_STATE CurrentState = STATE_BOOT;
CTrade        Trade;

//--- Symbol registry
string   Symbols[];
int      SymbolCount = 0;

//--- Per-symbol indicator handles
int      ATR_Handles[];

//--- Per-symbol SMC brain state
datetime LastBarTime[];        // H1 bar time — gates ATR trail (candle-gated)
datetime StagnationBarTime[];  // H1 bar time — gates stagnation counter
int      StagnationCount[];    // Consecutive bars without a new extreme
ulong    StagnationTicket[];   // Ticket bound to the current stagnation counter

//--- Per-symbol signal deduplication
string   LastSignalHash[];

//--- Per-symbol throttled log for legacy IR= missing warning (once/minute)
datetime IRMissingLastLog[];

//--- Per-symbol bot-closed ticket guard
//    When the bot sends POSITION_CLOSED and we close the trade, we record the
//    ticket here so OnTick skips management while the broker clears the position.
//    Cleared when a new trade is opened on the same symbol slot.
ulong    BotClosedTickets[];

//--- State-machine timing
datetime NextPollTime       = 0;
datetime NextEventPollTime  = 0;
datetime NextDataPushTime   = 0;
datetime RetryTime          = 0;

//==========================================================================
//  OnInit
//==========================================================================
int OnInit() {
   if(StringLen(ApiBaseUrl) < 10) {
      Print("[INIT][ERROR] ApiBaseUrl is too short or empty.");
      return INIT_PARAMETERS_INCORRECT;
   }

   SymbolCount = ParseSymbolList(SymbolList, Symbols);
   if(SymbolCount < 1 || SymbolCount > 8) {
      PrintFormat("[INIT][ERROR] Expected 1-8 symbols, got %d.", SymbolCount);
      return INIT_PARAMETERS_INCORRECT;
   }

   //--- Allocate all per-symbol arrays
   ArrayResize(ATR_Handles,      SymbolCount);
   ArrayResize(LastBarTime,      SymbolCount);
   ArrayResize(StagnationBarTime,SymbolCount);
   ArrayResize(StagnationCount,  SymbolCount);
   ArrayResize(StagnationTicket, SymbolCount);
   ArrayResize(LastSignalHash,   SymbolCount);
   ArrayResize(IRMissingLastLog, SymbolCount);
   ArrayResize(BotClosedTickets, SymbolCount);

   for(int i = 0; i < SymbolCount; i++) {
      if(!SymbolSelect(Symbols[i], true))
         PrintFormat("[INIT][WARN] %s not in Market Watch — attempting to add.", Symbols[i]);

      ATR_Handles[i] = iATR(Symbols[i], PERIOD_H1, ATR_Period);
      if(ATR_Handles[i] == INVALID_HANDLE) {
         PrintFormat("[INIT][ERROR] Failed to create ATR handle for %s.", Symbols[i]);
         return INIT_FAILED;
      }

      LastBarTime[i]       = 0;
      StagnationBarTime[i] = 0;
      StagnationCount[i]   = 0;
      StagnationTicket[i]  = 0;
      LastSignalHash[i]    = "";
      IRMissingLastLog[i]  = 0;
      BotClosedTickets[i]  = 0;

      PrintFormat("[INIT] Slot %d -> %s | ATR handle: %d", i, Symbols[i], ATR_Handles[i]);
   }

   Trade.SetExpertMagicNumber(MagicNumber);
   Trade.SetDeviationInPoints(SlippagePoints);

   EventSetTimer(1);
   CurrentState = STATE_BOOT;

   PrintFormat("[INIT] FxGuru_SMC_Hub v3.00 | %d symbol(s) | Magic: %d | Fund filter: %s (%s)",
               SymbolCount, MagicNumber,
               FundamentalFilterEnabled ? "ON" : "OFF",
               FundamentalFilterMode);
   return INIT_SUCCEEDED;
}

//==========================================================================
//  OnDeinit
//==========================================================================
void OnDeinit(const int reason) {
   EventKillTimer();
   for(int i = 0; i < SymbolCount; i++) {
      if(ATR_Handles[i] != INVALID_HANDLE)
         IndicatorRelease(ATR_Handles[i]);
   }
   PrintFormat("[DEINIT] Stopped. Reason: %d", reason);
}

//==========================================================================
//  OnTimer — State Machine Dispatcher
//==========================================================================
void OnTimer() {
   switch(CurrentState) {

      case STATE_BOOT:
         PerformHandshake();
         break;

      case STATE_POLLING:
         //--- Data push takes priority; breaks out immediately to avoid
         //    polling while a large POST is already pending
         if(TimeCurrent() >= NextDataPushTime) {
            CurrentState = STATE_DATA_PUSH;
            break;
         }

         //--- Run event and signal polls in the correct dependency order
         if(TimeCurrent() >= NextEventPollTime || TimeCurrent() >= NextPollTime) {
            bool doEvents  = (TimeCurrent() >= NextEventPollTime);
            bool doSignals = (TimeCurrent() >= NextPollTime);

            //--- Position events FIRST: reconcile state before acting on signals
            if(doEvents) {
               for(int i = 0; i < SymbolCount; i++)
                  PollForPositionEvents(i);
               NextEventPollTime = (datetime)(TimeCurrent() + EventPollIntervalSeconds);
            }

            //--- Entry signals SECOND
            if(doSignals) {
               for(int i = 0; i < SymbolCount; i++)
                  PollForSignals(i);
               NextPollTime = (datetime)(TimeCurrent() + PollIntervalSeconds);
            }
         }
         break;

      case STATE_DATA_PUSH:
         for(int i = 0; i < SymbolCount; i++)
            PushMarketData(i);
         NextDataPushTime = (datetime)(TimeCurrent() + ((long)DataPushMinutes * 60));
         NextPollTime      = TimeCurrent();
         NextEventPollTime = TimeCurrent();
         CurrentState      = STATE_POLLING;
         break;

      case STATE_ERROR_RECOVERY:
         if(RetryTime == 0) {
            Print("[RECOVERY] Connection lost. Retrying in 15 seconds...");
            RetryTime = TimeCurrent() + 15;
         }
         if(TimeCurrent() >= RetryTime) {
            RetryTime    = 0;
            CurrentState = STATE_BOOT;
         }
         break;

      default:
         break;
   }
}

//==========================================================================
//  OnTick — Management Brain Dispatcher
//  Runs every tick so Break-Even reacts to price instantly.
//  Skips positions whose ticket was just closed by a bot event.
//==========================================================================
void OnTick() {
   for(int i = 0; i < SymbolCount; i++) {
      if(!SelectPositionByMagicAndSymbol(MagicNumber, Symbols[i])) continue;

      ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);

      //--- Guard: do not manage a position we just closed via bot event
      //    BotClosedTickets[i] remains set until a new trade opens on this slot
      if(BotClosedTickets[i] != 0 && BotClosedTickets[i] == ticket) {
         if(DebugLogs)
            PrintFormat("[TICK][%s] Skipping management — ticket %I64u closing via bot event.",
                        Symbols[i], ticket);
         continue;
      }

      ManageOpenPosition(i);
   }
}

//==========================================================================
//  FEATURE 1 — MARKET DATA PUSH  (Twelve Data Emulator)
//  POST /update-data/{symbol}
//  Schema: { "symbol": "X", "timeframes": { "H1": [...100 bars...],
//                                           "H4": [...100 bars...] } }
//  Candle fields: t (unix timestamp), o, h, l, c, v — matches server schema.
//==========================================================================
void PushMarketData(int idx) {
   string sym = Symbols[idx];
   string url = ApiBaseUrl + "/update-data/" + sym;

   string payload = "{";
   payload += "\"symbol\":\"" + sym + "\",";
   payload += "\"timeframes\":{";
   payload += "\"H1\":"  + BuildCandleJSON(sym, PERIOD_H1, 100) + ",";
   payload += "\"H4\":"  + BuildCandleJSON(sym, PERIOD_H4, 100);
   payload += "}}";

   string response;
   int code = SendPostRequest(url, payload, response);

   if(code == 200 || code == 201 || code == 204) {
      if(DebugLogs) PrintFormat("[DATA][%s] Push OK -> %s", sym, response);
   } else {
      PrintFormat("[DATA][%s][WARN] Push failed. HTTP %d | %s", sym, code, response);
   }
}

//--- Serialise N closed candles. Skips bar[0] (live/incomplete). Oldest first.
//    Short field names match the server's ingest schema: t, o, h, l, c, v.
string BuildCandleJSON(string sym, ENUM_TIMEFRAMES tf, int count) {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(sym, tf, 1, count, rates);
   if(copied <= 0) {
      PrintFormat("[DATA][%s] CopyRates returned %d for %s.", sym, copied, EnumToString(tf));
      return "[]";
   }

   string json = "[";
   for(int i = copied - 1; i >= 0; i--) {
      json += StringFormat(
         "{\"t\":%d,\"o\":%.5f,\"h\":%.5f,\"l\":%.5f,\"c\":%.5f,\"v\":%d}",
         (long)rates[i].time,
         rates[i].open, rates[i].high, rates[i].low, rates[i].close,
         (long)rates[i].tick_volume
      );
      if(i > 0) json += ",";
   }
   return json + "]";
}

//==========================================================================
//  FEATURE 2a — POSITION EVENT POLLING
//  GET /poll/position-events/{symbol}
//
//  Responses:
//    {"status":"no_event"}
//    {"status":"event","event_type":"POSITION_CLOSED",
//     "ticket":12345,"reason":"signal_invalid","exit_price":1.0882,"pnl":-12.50}
//==========================================================================
void PollForPositionEvents(int idx) {
   string sym    = Symbols[idx];
   string result;
   int    code   = SendGetRequest(ApiBaseUrl + "/poll/position-events/" + sym, result);

   if(code != 200) {
      if(DebugLogs) PrintFormat("[EVENT][%s] HTTP %d", sym, code);
      return;
   }
   if(DebugLogs) PrintFormat("[EVENT][%s] %s", sym, result);

   if(StringFind(result, "\"no_event\"") >= 0) return;

   if(StringFind(result, "\"event\"")          >= 0 &&
      StringFind(result, "\"POSITION_CLOSED\"") >= 0)
      HandlePositionClosedEvent(idx, result);
}

//--- Execute a POSITION_CLOSED event received from the bot.
//    The bot has determined the original signal is no longer valid.
//    We close the matching MT5 position and reset symbol state.
void HandlePositionClosedEvent(int idx, string json) {
   string sym = Symbols[idx];

   //--- Parse optional event fields
   string botTicketStr = GetJsonValue(json, "ticket");
   string reason       = GetJsonValue(json, "reason");
   string exitPriceStr = GetJsonValue(json, "exit_price");
   string pnlStr       = GetJsonValue(json, "pnl");

   ulong  botTicket = (ulong)(StringLen(botTicketStr) > 0 ? StringToInteger(botTicketStr) : 0);
   double exitPrice = (StringLen(exitPriceStr) > 0) ? StringToDouble(exitPriceStr) : 0.0;
   double pnl       = (StringLen(pnlStr)       > 0) ? StringToDouble(pnlStr)       : 0.0;

   PrintFormat("[EVENT][%s] POSITION_CLOSED | Reason: %s | Exit: %.5f | PnL: %.2f | BotTicket: %I64u",
               sym, reason, exitPrice, pnl, botTicket);

   //--- Find the local open position for this symbol
   if(!SelectPositionByMagicAndSymbol(MagicNumber, sym)) {
      PrintFormat("[EVENT][%s] No matching position found — may already be closed.", sym);
      return;
   }

   ulong  localTicket = (ulong)PositionGetInteger(POSITION_TICKET);
   double localPrice  = PositionGetDouble(POSITION_PRICE_CURRENT);

   //--- Register ticket BEFORE attempting close so OnTick skips it immediately
   BotClosedTickets[idx] = localTicket;

   if(Trade.PositionClose(sym)) {
      PrintFormat("[EVENT][%s] Ticket %I64u closed by bot event. Reason: %s",
                  sym, localTicket, reason);

      //--- Clear SMC brain state for this symbol slot
      StagnationCount[idx]   = 0;
      StagnationTicket[idx]  = 0;
      StagnationBarTime[idx] = 0;
      LastBarTime[idx]       = 0;

      //--- Annotate chart for post-trade review (visible in MT5 chart comments)
      ChartSetString(0, CHART_COMMENT,
         StringFormat("BOT CLOSE [%s]: %s | Price: %.5f | PnL: %.2f",
                      sym, reason, localPrice, pnl));

      //--- Clear dedup hash: bot may re-evaluate and send a fresh signal shortly
      LastSignalHash[idx] = "";

   } else {
      //--- Close failed — clear ticket guard so management can continue
      BotClosedTickets[idx] = 0;
      PrintFormat("[EVENT][%s] CLOSE FAILED. Retcode: %d | %s",
                  sym, Trade.ResultRetcode(), Trade.ResultRetcodeDescription());
   }
}

//==========================================================================
//  FEATURE 2b — ENTRY SIGNAL POLLING
//  GET /poll/{symbol}
//==========================================================================
void PollForSignals(int idx) {
   string sym    = Symbols[idx];
   string result;
   int    code   = SendGetRequest(ApiBaseUrl + "/poll/" + sym, result);

   if(code != 200) {
      PrintFormat("[POLL][%s] HTTP %d", sym, code);
      return;
   }
   if(DebugLogs) PrintFormat("[DEBUG][%s] %s", sym, result);

   if(StringFind(result, "\"no_signal\"") >= 0) return;
   if(StringFind(result, "\"expired\"") >= 0) {
      if(DebugLogs) PrintFormat("[POLL][%s] Signal expired.", sym);
      return;
   }
   if(StringFind(result, "\"new_trade\"") >= 0)
      ExecuteTradeFromJSON(idx, result);
}

//==========================================================================
//  FEATURE 2c — TRADE EXECUTION
//  Parses the full /poll signal payload including fundamental analyst fields.
//
//  Full expected JSON (all fields flat — no nesting required for parsing):
//  {
//    "status":                 "new_trade",
//    "direction":              "BUY",
//    "entry":                  1.08450,
//    "sl":                     1.08100,
//    "tp":                     1.09100,
//    "fundamental_rating":     1,          <- int: -1 | 0 | 1 (absent = 0)
//    "fundamental_conviction": "strong",   <- "weak"|"moderate"|"strong"
//    "fundamental_note":       "..."       <- plain text summary string
//  }
//
//  Backward compatible: all three fundamental fields are optional.
//  Absent fields default to rating=0 (neutral) so no filter action fires.
//==========================================================================
void ExecuteTradeFromJSON(int idx, string json) {
   string sym = Symbols[idx];

   //--- Core signal fields
   string dir   = GetJsonValue(json, "direction");
   double entry = StringToDouble(GetJsonValue(json, "entry"));
   double sl    = StringToDouble(GetJsonValue(json, "sl"));
   double tp    = StringToDouble(GetJsonValue(json, "tp"));

   //--- Fundamental analyst fields (optional, default-safe)
   int    fundamentalRating     = 0;
   string fundamentalConviction = "weak";
   string fundamentalNote       = "";

   string ratingStr = GetJsonValue(json, "fundamental_rating");
   if(StringLen(ratingStr) > 0)
      fundamentalRating = (int)StringToInteger(ratingStr);

   string convStr = GetJsonValue(json, "fundamental_conviction");
   if(StringLen(convStr) > 0)
      fundamentalConviction = convStr;

   string noteStr = GetJsonValue(json, "fundamental_note");
   if(StringLen(noteStr) > 0)
      fundamentalNote = noteStr;

   //--- Always log the fundamental context for audit trail (all filter modes)
   PrintFormat("[FUNDAMENTAL][%s] Rating: %+d | Conviction: %s | %s",
               sym, fundamentalRating, fundamentalConviction,
               StringLen(fundamentalNote) > 0 ? fundamentalNote : "(no note)");

   //--- Deduplication
   string hash = dir + DoubleToString(entry, 5);
   if(hash == LastSignalHash[idx]) {
      if(DebugLogs) PrintFormat("[EXEC][%s] Duplicate signal suppressed.", sym);
      return;
   }

   //--- Direction mapping
   ENUM_ORDER_TYPE orderType = WRONG_VALUE;
   if(dir == "BUY"  || dir == "LONG")  { if(!AllowLong)  return; orderType = ORDER_TYPE_BUY;  }
   if(dir == "SELL" || dir == "SHORT") { if(!AllowShort) return; orderType = ORDER_TYPE_SELL; }
   if(orderType == WRONG_VALUE) {
      PrintFormat("[EXEC][%s][ERROR] Unknown direction: '%s'", sym, dir);
      return;
   }

   //--- Pre-trade checks
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) {
      Print("[EXEC] AutoTrading is disabled in terminal.");
      return;
   }
   double spread = (double)SymbolInfoInteger(sym, SYMBOL_SPREAD);
   if(spread > MaxSpreadPoints) {
      PrintFormat("[EXEC][%s] Spread %.0f > max %.0f. Skipping.", sym, spread, MaxSpreadPoints);
      return;
   }
   if(OnePositionPerSymbol && SelectPositionByMagicAndSymbol(MagicNumber, sym)) {
      PrintFormat("[EXEC][%s] Position already open. Skipping.", sym);
      return;
   }

   //--- Execution price and base lot size
   double price = (orderType == ORDER_TYPE_BUY)
                  ? SymbolInfoDouble(sym, SYMBOL_ASK)
                  : SymbolInfoDouble(sym, SYMBOL_BID);

   if(!UseDynamicSLTP) { sl = 0.0; tp = 0.0; }

   double initialRiskPts = (sl > 0.0) ? MathAbs(price - sl) : 0.0;
   double volume = (initialRiskPts > 0.0 && UseDynamicSLTP)
                   ? CalculateLotSize(sym, price, sl)
                   : FixedLotFallback;

   //--- ── Fundamental Filter ─────────────────────────────────────────────
   //    Executes AFTER lot calculation so SOFT mode can halve the computed lot.
   //    Only activates on rating = -1 (opposition).
   //    Rating = 0 (neutral) and +1 (support) always trade at full lot.
   if(FundamentalFilterEnabled && fundamentalRating == -1) {

      int convictionLevel = 0;
      if(fundamentalConviction == "moderate") convictionLevel = 1;
      if(fundamentalConviction == "strong")   convictionLevel = 2;

      bool hardBlockApplies = (FundamentalFilterMode == "HARD" &&
                               convictionLevel >= FundamentalHardMinConviction);
      bool softApplies      = (FundamentalFilterMode == "SOFT") ||
                              (FundamentalFilterMode == "HARD" &&
                               convictionLevel < FundamentalHardMinConviction);

      if(hardBlockApplies) {
         //--- HARD BLOCK: reject signal entirely, do not open position
         PrintFormat("[FUNDAMENTAL][%s] HARD BLOCK — rejected. "
                     "Rating: -1 | Conviction: %s | %s",
                     sym, fundamentalConviction, fundamentalNote);
         return;
      }

      if(softApplies) {
         //--- SOFT REDUCE: halve lot size to express headwind caution.
         //    Also activates when HARD mode is set but conviction is below
         //    FundamentalHardMinConviction (treats weak opposition as SOFT).
         double minVol  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
         double step    = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
         double halfVol = MathFloor(volume * 0.5 / step) * step;
         volume = MathMax(minVol, halfVol);

         PrintFormat("[FUNDAMENTAL][%s] SOFT REDUCE — lot -> %.2f. "
                     "Rating: -1 | Conviction: %s | %s",
                     sym, volume, fundamentalConviction, fundamentalNote);
      }
      //--- FundamentalFilterMode == "OFF": fall through, no adjustment
   }
   //--- ── End Fundamental Filter ─────────────────────────────────────────

   //--- Encode persistent management data into the position comment
   string comment = StringFormat("IR=%.5f|IV=%.2f", initialRiskPts, volume);

   PrintFormat("[EXEC][%s] %s | Vol: %.2f | SL: %.5f | TP: %.5f | %s | Fund: %+d/%s",
               sym, dir, volume, sl, tp, comment, fundamentalRating, fundamentalConviction);

   if(Trade.PositionOpen(sym, orderType, volume, price, sl, tp, comment)) {
      PrintFormat("[EXEC][%s] Trade opened. Ticket: %I64u", sym, Trade.ResultOrder());
      LastSignalHash[idx]    = hash;
      StagnationCount[idx]   = 0;
      StagnationTicket[idx]  = Trade.ResultOrder();
      StagnationBarTime[idx] = 0;
      BotClosedTickets[idx]  = 0;  // Clear any stale close guard for this symbol slot
   } else {
      PrintFormat("[EXEC][%s] FAILED. Retcode: %d | %s",
                  sym, Trade.ResultRetcode(), Trade.ResultRetcodeDescription());
   }
}

//--- Dynamic 1% risk lot calculator
double CalculateLotSize(string sym, double entry, double sl) {
   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * (RiskPercent / 100.0);
   double tickVal   = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double slDist    = MathAbs(entry - sl);

   if(slDist <= 0.0 || tickVal <= 0.0 || tickSize <= 0.0) {
      PrintFormat("[RISK][%s] Degenerate values — using fallback lot.", sym);
      return FixedLotFallback;
   }

   double points  = slDist / tickSize;
   double rawLots = riskMoney / (points * tickVal);

   double step   = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);

   return MathMax(minLot, MathMin(maxLot, MathFloor(rawLots / step) * step));
}

//==========================================================================
//  FEATURE 3 — SMC MANAGEMENT BRAIN
//
//  Design invariants:
//    * R-multiple uses stored IR= from comment — never live SL (immune to
//      SL migration corruption: once SL = entry, live distance ~ 0).
//    * SL only moves in the direction of better protection (one-directional).
//    * BE must be confirmed before Partial TP fires (enforced dependency).
//    * Trail and Stagnation are candle-gated to resist tick-level noise.
//    * Legacy positions (no IR= in comment) use fallback reconstruction
//      from the stored SL, logged at throttled rate (once per minute).
//==========================================================================
void ManageOpenPosition(int idx) {
   string sym     = Symbols[idx];
   string comment = PositionGetString(POSITION_COMMENT);

   //--- Primary: recover risk from comment (restart-proof)
   double initialRisk = ExtractCommentValue(comment, "IR");
   double initialVol  = ExtractCommentValue(comment, "IV");

   //--- ── Legacy Fallback ───────────────────────────────────────────────
   //    For positions opened by the old EA (comment = "API Signal") or
   //    any trade opened before IR= encoding was introduced.
   //    Reconstruction is valid ONLY while the SL has not yet been migrated.
   //    After Stage 1 fires and SL = entry, initialRisk ≈ 0 → management
   //    halts safely (rather than producing infinite R-multiples).
   //    Resolution: close and re-enter via this EA so IR= is encoded.
   if(initialRisk <= 0.0) {
      double legacyEntry = PositionGetDouble(POSITION_PRICE_OPEN);
      double legacySL    = PositionGetDouble(POSITION_SL);

      if(legacySL > 0.0 && legacyEntry > 0.0) {
         initialRisk = MathAbs(legacyEntry - legacySL);
         initialVol  = PositionGetDouble(POSITION_VOLUME);

         if(TimeCurrent() - IRMissingLastLog[idx] >= 60) {
            PrintFormat("[BRAIN][%s][LEGACY] No IR= in comment '%s'. "
                        "Reconstructed IR=%.5f from SL. NOT restart-proof. "
                        "Re-enter via new EA to fix permanently.",
                        sym, comment, initialRisk);
            IRMissingLastLog[idx] = TimeCurrent();
         }
      } else {
         if(TimeCurrent() - IRMissingLastLog[idx] >= 60) {
            PrintFormat("[BRAIN][%s][WARN] Comment '%s' has no IR= and position "
                        "has no SL. Cannot manage safely — set SL manually or re-enter.",
                        sym, comment);
            IRMissingLastLog[idx] = TimeCurrent();
         }
         return;
      }
   }

   //--- Position snapshot
   double entry      = PositionGetDouble(POSITION_PRICE_OPEN);
   double currentSL  = PositionGetDouble(POSITION_SL);
   double currentTP  = PositionGetDouble(POSITION_TP);
   double currentVol = PositionGetDouble(POSITION_VOLUME);
   long   posType    = PositionGetInteger(POSITION_TYPE);
   ulong  ticket     = (ulong)PositionGetInteger(POSITION_TICKET);
   int    digits     = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);

   //--- R-multiple from STORED initial risk — immune to SL migration
   double rMultiple = (posType == POSITION_TYPE_BUY)
                      ? (bid - entry) / initialRisk
                      : (entry - ask) / initialRisk;

   //--- State flags derived from live position data
   bool beConfirmed = (posType == POSITION_TYPE_BUY)
                      ? (currentSL >= entry)
                      : (currentSL > 0.0 && currentSL <= entry);

   bool partialDone = (initialVol > 0.0) && (currentVol < initialVol * 0.75);

   //--- Broker-agnostic pip size (5-digit FX, 3-digit JPY/XAU)
   double pipSize = SymbolInfoDouble(sym, SYMBOL_POINT) * 10.0;
   double bePip   = SlippagePips * pipSize;

   // ===================================================================
   //  STAGE 1 — Break-Even  (tick-accurate, one-shot)
   //
   //  WHY 1.0R not 0.5R:
   //    Below 1.0R the trade is inside normal H1 noise. A premature BE
   //    at 0.5R gets hunted by the very next retracement wick in ~40%
   //    of cases. 1.0R gives the trade one full consolidation before
   //    protection is locked.
   // ===================================================================
   if(!beConfirmed && rMultiple >= BE_Trigger_R) {
      double bePrice = (posType == POSITION_TYPE_BUY)
                       ? NormalizeDouble(entry + bePip, digits)
                       : NormalizeDouble(entry - bePip, digits);

      bool shouldMove = (posType == POSITION_TYPE_BUY)
                        ? (bePrice > currentSL)
                        : (currentSL == 0.0 || bePrice < currentSL);

      if(shouldMove) {
         if(Trade.PositionModify(sym, bePrice, currentTP)) {
            PrintFormat("[BRAIN][%s] Stage 1: BE set -> %.5f (%.2fR)", sym, bePrice, rMultiple);
            currentSL   = bePrice;
            beConfirmed = true;
         } else {
            PrintFormat("[BRAIN][%s] Stage 1 FAILED. Retcode: %d", sym, Trade.ResultRetcode());
         }
      }
   }

   // ===================================================================
   //  STAGE 2 — Partial Take-Profit  (one-shot at Partial_TP_R)
   //
   //  Dependency: beConfirmed MUST be true first.
   //    If a fast move skips 1.0R and hits 1.5R in one tick, Stage 1
   //    fires on the same iteration (above) setting beConfirmed = true,
   //    then Stage 2 fires immediately after. The runner is never left
   //    with the original SL after a scale-out.
   // ===================================================================
   if(beConfirmed && !partialDone && rMultiple >= Partial_TP_R) {
      double step     = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
      double minVol   = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
      double closeVol = MathFloor(currentVol * 0.5 / step) * step;

      if(closeVol >= minVol) {
         if(Trade.PositionClosePartial(sym, closeVol)) {
            PrintFormat("[BRAIN][%s] Stage 2: Closed 50%% (%.2f lots) at %.2fR",
                        sym, closeVol, rMultiple);
            partialDone = true;
         } else {
            PrintFormat("[BRAIN][%s] Stage 2 FAILED. Retcode: %d", sym, Trade.ResultRetcode());
         }
      } else {
         if(DebugLogs)
            PrintFormat("[BRAIN][%s] Stage 2 skipped: closeVol %.2f < minVol %.2f",
                        sym, closeVol, minVol);
      }
   }

   // ===================================================================
   //  STAGE 3 — Structural ATR Trailing Stop  (candle-gated)
   //
   //  Candle-gate: bar[0] low shifts with intra-candle wicks. Only
   //  confirmed closed bars produce stable swing-low readings. Updating
   //  on every tick would move the SL based on incomplete candle data.
   //
   //  Post-partial tightening: after 50% is banked the runner is risk-free
   //  (SL >= entry), so we tighten to ATR_Tight_Mult to lock in gains more
   //  aggressively without risking the original capital.
   // ===================================================================
   if(beConfirmed) {
      datetime currentBarOpen = iTime(sym, PERIOD_H1, 0);
      if(currentBarOpen == 0) return;

      if(currentBarOpen != LastBarTime[idx]) {
         LastBarTime[idx] = currentBarOpen;

         double atr = GetATRValue(idx);
         if(atr <= 0.0) {
            if(DebugLogs) PrintFormat("[BRAIN][%s] Stage 3: ATR not ready.", sym);
         } else {
            double mult  = partialDone ? ATR_Tight_Mult : ATR_Buffer_Mult;
            double newSL = 0.0;

            if(posType == POSITION_TYPE_BUY) {
               int    lowestBar = iLowest(sym, PERIOD_H1, MODE_LOW, Swing_Lookback, 1);
               double swingLow  = iLow(sym, PERIOD_H1, lowestBar);
               newSL = NormalizeDouble(swingLow - (mult * atr), digits);

               //--- One-directional guard: only advance SL upward, never below entry
               if(newSL > currentSL && newSL >= entry) {
                  if(Trade.PositionModify(sym, newSL, currentTP)) {
                     PrintFormat("[BRAIN][%s] Stage 3 BUY -> %.5f (swing: %.5f ATR: %.5f x%.2f)",
                                 sym, newSL, swingLow, atr, mult);
                     currentSL = newSL;
                  }
               }
            } else {
               int    highestBar = iHighest(sym, PERIOD_H1, MODE_HIGH, Swing_Lookback, 1);
               double swingHigh  = iHigh(sym, PERIOD_H1, highestBar);
               newSL = NormalizeDouble(swingHigh + (mult * atr), digits);

               //--- One-directional guard: only advance SL downward for sells
               if(currentSL == 0.0 || newSL < currentSL) {
                  if(Trade.PositionModify(sym, newSL, currentTP)) {
                     PrintFormat("[BRAIN][%s] Stage 3 SELL -> %.5f (swing: %.5f ATR: %.5f x%.2f)",
                                 sym, newSL, swingHigh, atr, mult);
                     currentSL = newSL;
                  }
               }
            }
         }
      }
   }

   // ===================================================================
   //  STAGE 4 — Stagnation Exit  (candle-gated)
   //
   //  WHY dual-condition (R-level AND time counter):
   //    Pure time-exit would close a healthy trade in normal flag
   //    consolidation before a continuation move. Requiring rMultiple
   //    >= Stagnation_R ensures we only trigger on stagnation that is
   //    genuinely going nowhere further profitable. If price drops below
   //    Stagnation_R the ATR trail governs and the timer resets.
   //
   //  Reset: any bar printing a new extreme resets the counter,
   //  preserving flag/pennant patterns preceding continuation moves.
   // ===================================================================
   if(beConfirmed && rMultiple >= Stagnation_R) {
      datetime currentBarOpen = iTime(sym, PERIOD_H1, 0);
      if(currentBarOpen == 0) return;

      //--- Bind counter to current ticket (auto-resets when a new trade opens)
      if(StagnationTicket[idx] != ticket) {
         StagnationCount[idx]   = 0;
         StagnationTicket[idx]  = ticket;
         StagnationBarTime[idx] = 0;
      }

      if(currentBarOpen != StagnationBarTime[idx]) {
         StagnationBarTime[idx] = currentBarOpen;

         double bar1High = iHigh(sym, PERIOD_H1, 1);
         double bar1Low  = iLow(sym,  PERIOD_H1, 1);
         double bar2High = iHigh(sym, PERIOD_H1, 2);
         double bar2Low  = iLow(sym,  PERIOD_H1, 2);

         bool madeNewExtreme = (posType == POSITION_TYPE_BUY)
                               ? (bar1High > bar2High)
                               : (bar1Low  < bar2Low);

         if(madeNewExtreme) {
            StagnationCount[idx] = 0;
            if(DebugLogs)
               PrintFormat("[BRAIN][%s] Stage 4: Progress detected — counter reset.", sym);
         } else {
            StagnationCount[idx]++;
            PrintFormat("[BRAIN][%s] Stage 4: Stagnation %d/%d at %.2fR",
                        sym, StagnationCount[idx], Stagnation_Candles, rMultiple);
         }

         if(StagnationCount[idx] >= Stagnation_Candles) {
            PrintFormat("[BRAIN][%s] Stage 4: STAGNATION EXIT — %d candles, %.2fR.",
                        sym, Stagnation_Candles, rMultiple);
            if(Trade.PositionClose(sym)) {
               PrintFormat("[BRAIN][%s] Closed by stagnation exit.", sym);
               StagnationCount[idx]   = 0;
               StagnationTicket[idx]  = 0;
               StagnationBarTime[idx] = 0;
            } else {
               PrintFormat("[BRAIN][%s] Stagnation close FAILED. Retcode: %d",
                           sym, Trade.ResultRetcode());
            }
            return;
         }
      }
   } else {
      //--- Below stagnation floor: reset counter to prevent stale accumulation
      if(StagnationTicket[idx] == ticket)
         StagnationCount[idx] = 0;
   }
}

//==========================================================================
//  HELPERS — Position Selection
//  MQL5 has no native PositionSelectByMagic(). We iterate and match manually.
//  PositionSelectByTicket() sets the implicit selection used by all
//  subsequent PositionGet*() calls within the calling function scope.
//==========================================================================
bool SelectPositionByMagicAndSymbol(long magic, string sym) {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) == magic &&
         PositionGetString(POSITION_SYMBOL)  == sym)
         return true;
   }
   return false;
}

//==========================================================================
//  HELPERS — Indicators & Utilities
//==========================================================================

//--- Read last CLOSED bar ATR from pre-created handle (bar index 1).
//    Bar 0 is the live/incomplete candle — excluded to avoid noise.
double GetATRValue(int idx) {
   double buf[1];
   if(CopyBuffer(ATR_Handles[idx], 0, 1, 1, buf) <= 0) return 0.0;
   return buf[0];
}

//--- Extract a named float from a pipe-delimited comment string "IR=x|IV=y"
double ExtractCommentValue(string comment, string key) {
   string search = key + "=";
   int pos = StringFind(comment, search);
   if(pos < 0) return 0.0;
   int start = pos + StringLen(search);
   int end   = StringFind(comment, "|", start);
   if(end < 0) end = StringLen(comment);
   return StringToDouble(StringSubstr(comment, start, end - start));
}

//--- Parse "EURUSD,GBPUSD,USDJPY" into a string array. Max 8 elements.
int ParseSymbolList(string list, string &out[]) {
   StringReplace(list, " ", "");
   int count = 0;
   while(StringLen(list) > 0 && count < 8) {
      int    comma = StringFind(list, ",");
      string token;
      if(comma >= 0) {
         token = StringSubstr(list, 0, comma);
         list  = StringSubstr(list, comma + 1);
      } else {
         token = list;
         list  = "";
      }
      if(StringLen(token) > 0) {
         ArrayResize(out, count + 1);
         out[count++] = token;
      }
   }
   return count;
}

//==========================================================================
//  FEATURE 4 — NETWORKING
//==========================================================================

//--- /health handshake. Detects Error 4060 (URL not whitelisted in MT5).
void PerformHandshake() {
   CurrentState = STATE_HANDSHAKE_PENDING;
   string response;
   int    code = SendGetRequest(ApiBaseUrl + "/health", response);

   if(code == 4060) {
      Print("*==========================================================*");
      Print("  CRITICAL: Error 4060 — URL not whitelisted in MT5         ");
      Print("  Tools -> Options -> Expert Advisors -> Allowed URLs        ");
      PrintFormat("  Add: %s", ApiBaseUrl);
      Print("*==========================================================*");
      CurrentState = STATE_ERROR_RECOVERY;
      return;
   }

   if(code == 200) {
      string clean = response;
      StringReplace(clean, " ", "");
      StringReplace(clean, "\n", "");
      StringReplace(clean, "\"", "");

      if(StringFind(clean, "ok") >= 0 || StringFind(clean, "data:ok") >= 0) {
         Print("[HANDSHAKE] Server online -> STATE_POLLING");
         CurrentState      = STATE_POLLING;
         NextPollTime      = TimeCurrent();
         NextEventPollTime = TimeCurrent();
         NextDataPushTime  = TimeCurrent();
      } else {
         PrintFormat("[HANDSHAKE] Unexpected response: %s", response);
         CurrentState = STATE_ERROR_RECOVERY;
      }
   } else {
      PrintFormat("[HANDSHAKE] HTTP %d. Check Render URL and network.", code);
      CurrentState = STATE_ERROR_RECOVERY;
   }
}

//--- GET helper. Returns HTTP status code, or 4060 on whitelist error.
int SendGetRequest(string url, string &response) {
   char   data[], res[];
   string headers = "Accept: application/json\r\n";
   if(EnableApiKeyHeader && StringLen(ApiKey) > 0)
      headers += "X-API-KEY: " + ApiKey + "\r\n";

   int ret = WebRequest("GET", url, headers, 5000, data, res, response);
   if(ret == -1) {
      int err = GetLastError();
      if(err == 4060) return 4060;
      PrintFormat("[NET][GET] Error %d -> %s", err, url);
      return -1;
   }
   response = CharArrayToString(res);
   return ret;
}

//--- POST helper. 10 s timeout for large OHLCV payloads.
//    StringToCharArray with explicit count avoids the trailing null byte
//    that would corrupt the HTTP Content-Length sent to the server.
int SendPostRequest(string url, string payload, string &response) {
   char   data[], res[];
   StringToCharArray(payload, data, 0, StringLen(payload));

   string headers = "Content-Type: application/json\r\nAccept: application/json\r\n";
   if(EnableApiKeyHeader && StringLen(ApiKey) > 0)
      headers += "X-API-KEY: " + ApiKey + "\r\n";

   int ret = WebRequest("POST", url, headers, 10000, data, res, response);
   if(ret == -1) {
      int err = GetLastError();
      if(err == 4060) return 4060;
      PrintFormat("[NET][POST] Error %d -> %s", err, url);
      return -1;
   }
   response = CharArrayToString(res);
   return ret;
}

//--- Lightweight JSON value extractor (no external DLLs).
//    Handles both quoted strings and bare literals (numbers, booleans, ints).
//    Does not recurse into nested objects — sufficient for flat payloads.
string GetJsonValue(string json, string key) {
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0) return "";

   int start = pos + StringLen(pattern);

   //--- Skip optional whitespace after the colon
   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ')
      start++;

   bool isString = (StringGetCharacter(json, start) == '"');
   if(isString) start++;  // consume opening quote

   int end;
   if(isString) {
      end = StringFind(json, "\"", start);
      if(end < 0) end = StringLen(json);
   } else {
      end = StringLen(json);
      int c1 = StringFind(json, ",", start);
      int c2 = StringFind(json, "}", start);
      int c3 = StringFind(json, "]", start);
      if(c1 >= 0 && c1 < end) end = c1;
      if(c2 >= 0 && c2 < end) end = c2;
      if(c3 >= 0 && c3 < end) end = c3;
   }

   string val = StringSubstr(json, start, end - start);
   StringTrimLeft(val);
   StringTrimRight(val);
   return val;
}
//+------------------------------------------------------------------+
