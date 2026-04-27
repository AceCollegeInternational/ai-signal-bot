//+------------------------------------------------------------------+
//|                                              SignalDashboard.mq5 |
//|                                  Copyright 2024, Trading Bot     |
//|                                             https://example.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Trading Bot"
#property link      "https://example.com"
#property version   "1.00"
#property strict
#property indicator_chart_window

//--- Input parameters
input string   InpServerUrl = "http://localhost:8000/analysis/"; // Server URL
input int      InpPollInterval = 10;                            // Poll interval in seconds
input color    InpTextColor = clrWhite;                         // Text color
input int      InpFontSize = 10;                                // Font size

//--- Global variables
string         prefix = "AI_DASH_";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(InpPollInterval);
   CreateDashboard();
   PollAnalysis();
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectsDeleteAll(0, prefix);
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   PollAnalysis();
}

//+------------------------------------------------------------------+
//| Create Dashboard UI elements                                     |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   int x = 20;
   int y = 50;
   
   // Background Panel
   ObjectCreate(0, prefix+"BG", OBJ_RECTLABEL, 0, 0, 0);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_XDISTANCE, x-10);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_YDISTANCE, y-10);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_XSIZE, 400);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_YSIZE, 220);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_BGCOLOR, clrDarkSlateGray);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_BORDER_TYPE, BORDER_SUNKEN);
   
   // Title
   CreateLabel(prefix+"Title", x, y, "AI ANALYST SIGNAL", 12, clrCyan);
   
   // Signal Label
   CreateLabel(prefix+"Signal", x, y+30, "Signal: CACHE...", InpFontSize, InpTextColor);
   
   // Confidence Label
   CreateLabel(prefix+"Confidence", x, y+55, "Confidence: --%", InpFontSize, InpTextColor);
   
   // Reasoning Label (Multi-line approx)
   CreateLabel(prefix+"ReasoningTitle", x, y+90, "AI Reasoning:", 9, clrLightBlue);
   CreateLabel(prefix+"Reason1", x, y+110, "", 9, InpTextColor);
   CreateLabel(prefix+"Reason2", x, y+130, "", 9, InpTextColor);
   CreateLabel(prefix+"Reason3", x, y+150, "", 9, InpTextColor);
   
   // Age Label
   CreateLabel(prefix+"Age", x, y+185, "Last Update: --s ago", 8, clrGray);
}

void CreateLabel(string name, int x, int y, string text, int size, color col)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetString(0, name, OBJPROP_FONT, "Trebuchet MS");
}

//+------------------------------------------------------------------+
//| Poll server for analysis                                         |
//+------------------------------------------------------------------+
void PollAnalysis()
{
   string url = InpServerUrl + Symbol();
   char post[], result[];
   string headers;
   int timeout = 5000;
   
   int res = WebRequest("GET", url, NULL, timeout, post, result, headers);
   
   if(res == 200)
   {
      string response = CharArrayToString(result);
      UpdateUI(response);
   }
}

//+------------------------------------------------------------------+
//| Update UI with response data                                     |
//+------------------------------------------------------------------+
void UpdateUI(string json)
{
   // Basic parsing for MQL5 (Manual string extraction for the task)
   string signal = GetJsonField(json, "direction");
   string confidence = GetJsonField(json, "confidence");
   string reasoning = GetJsonField(json, "reasoning");
   string age = GetJsonField(json, "age_seconds");
   
   color sigColor = clrWhite;
   if(signal == "BUY" || signal == "LONG") sigColor = clrLime;
   else if(signal == "SELL" || signal == "SHORT") sigColor = clrRed;
   else sigColor = clrYellow;
   
   ObjectSetString(0, prefix+"Signal", OBJPROP_TEXT, "Signal: " + signal);
   ObjectSetInteger(0, prefix+"Signal", OBJPROP_COLOR, sigColor);
   
   ObjectSetString(0, prefix+"Confidence", OBJPROP_TEXT, "Confidence: " + confidence + "%");
   
   // Split reasoning into 3 lines (simplified wrap)
   ObjectSetString(0, prefix+"Reason1", OBJPROP_TEXT, StringSubstr(reasoning, 0, 60));
   ObjectSetString(0, prefix+"Reason2", OBJPROP_TEXT, StringSubstr(reasoning, 60, 60));
   ObjectSetString(0, prefix+"Reason3", OBJPROP_TEXT, StringSubstr(reasoning, 120, 60));
   
   ObjectSetString(0, prefix+"Age", OBJPROP_TEXT, "Last Update: " + age + "s ago");
   
   ChartRedraw();
}

string GetJsonField(string json, string field)
{
   string search = "\"" + field + "\":";
   int start = StringFind(json, search);
   if(start == -1) return "N/A";
   
   start += StringLen(search);
   
   // Check if it's a string value
   if(StringSubstr(json, start, 1) == "\"")
   {
      start++;
      int end = StringFind(json, "\"", start);
      return StringSubstr(json, start, end - start);
   }
   else
   {
      // Numeric or boolean value
      int end = StringFind(json, ",", start);
      if(end == -1) end = StringFind(json, "}", start);
      return StringSubstr(json, start, end - start);
   }
}
