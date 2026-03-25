#property script_show_inputs

input string InpSymbol = "EURUSD";
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input int InpBars = 12000;
input string InpSubdir = "tslib";

string TimeframeToString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1: return "M1";
      case PERIOD_M5: return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1: return "H1";
      case PERIOD_H4: return "H4";
      case PERIOD_D1: return "D1";
   }
   return "CUSTOM";
}

string BuildRow(datetime ts,
                double open_,
                double high_,
                double low_,
                double close_,
                long tick_volume,
                int spread,
                long real_volume,
                double prev_close)
{
   double range = high_ - low_;
   double body = close_ - open_;
   double ret1 = 0.0;
   double log_ret1 = 0.0;
   double hl_ratio = 0.0;

   if(prev_close > 0.0)
   {
      ret1 = (close_ - prev_close) / prev_close;
      log_ret1 = MathLog(close_ / prev_close);
   }
   if(close_ > 0.0)
      hl_ratio = range / close_;

   return StringFormat("%s,%.10f,%.10f,%.10f,%d,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f",
                       TimeToString(ts, TIME_DATE | TIME_MINUTES),
                       open_,
                       high_,
                       low_,
                       (int)tick_volume,
                       spread,
                       (int)real_volume,
                       range,
                       body,
                       ret1,
                       log_ret1,
                       hl_ratio,
                       close_);
}

void OnStart()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, false);

   int copied = CopyRates(InpSymbol, InpTimeframe, 0, InpBars, rates);
   if(copied <= 0)
   {
      Print("CopyRates failed: ", GetLastError());
      return;
   }

   string timeframe = TimeframeToString(InpTimeframe);
   string common_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH);
   string relative_path = InpSubdir + "\\" + InpSymbol + "_" + timeframe + ".csv";
   string full_path = common_path + "\\Files\\" + relative_path;

   int handle = FileOpen(relative_path, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("FileOpen failed: ", GetLastError(), " path=", full_path);
      return;
   }

   FileWriteString(handle, "date,open,high,low,tick_volume,spread,real_volume,range,body,return_1,log_return_1,hl_ratio,close\r\n");

   double prev_close = 0.0;
   for(int i = 0; i < copied; i++)
   {
      FileWriteString(
         handle,
         BuildRow(
            rates[i].time,
            rates[i].open,
            rates[i].high,
            rates[i].low,
            rates[i].close,
            rates[i].tick_volume,
            rates[i].spread,
            rates[i].real_volume,
            prev_close
         ) + "\r\n"
      );
      prev_close = rates[i].close;
   }

   FileClose(handle);
   Print("Exported ", copied, " bars to ", full_path);
}
