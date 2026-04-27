# Trading Bot Deployment Guide

This guide walks you through deploying your AI Trading Bot and Execution Server to a Windows VPS for 24/7 autonomous trading.

Because the bot relies on the `MetaTrader5` Python package (which requires the Windows MT5 terminal application), a standard Linux cloud server (like AWS EC2 Linux or Heroku) is not recommended. You must use a Windows environment.

---

## Phase 1: Securing a Windows VPS

1. **Choose a Provider:** Rent a Windows Virtual Private Server (VPS). Recommended options:
   - **Contabo:** Excellent value (e.g., Cloud VPS S with Windows Server 2022).
   - **ForexVPS:** Optimized specifically for MT5 with low latency to brokers.
   - **Vultr / DigitalOcean:** Standard cloud providers with Windows Server options.
2. **Server Specs:** Minimum 2 CPU Cores, 4GB RAM (8GB recommended for AI processing and MT5).
3. **Connect:** Use Remote Desktop Connection (RDP) on your local computer to log into the VPS using the IP address and Administrator credentials provided by your host.

---

## Phase 2: Installing Prerequisites on the VPS

Once connected to your VPS desktop:

1. **Install MetaTrader 5:**
   - Download the MT5 installer from your specific broker (e.g., Headway).
   - Install it and log into your trading account.
   - Go to `Tools -> Options -> Expert Advisors` and enable **"Allow algorithmic trading"**.
2. **Install Python:**
   - Download Python (3.10+ recommended) from [python.org](https://www.python.org/downloads/windows/).
   - ⚠️ **CRITICAL:** Check the box that says **"Add python.exe to PATH"** at the bottom of the installer before clicking "Install Now".
3. **Install Git:**
   - Download and install [Git for Windows](https://git-scm.com/download/win).
4. **Install Node.js (for PM2 Process Manager):**
   - Download and install [Node.js](https://nodejs.org/en/download/).

---

## Phase 3: Setting Up the Bot Code

1. Open PowerShell on the VPS.
2. Clone your code repository (or copy the files securely from your local machine):
   ```powershell
   cd C:\Users\Administrator\Desktop
   git clone https://github.com/yourusername/trading-bot.git
   cd "trading-bot"
   ```
3. Create and activate a Virtual Environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Configure your `.env` file:
   - Create or edit the `.env` file on the server.
   - Update the `MT5_TERMINAL_PATH` to point to the correct install location on the VPS (e.g., `C:\Program Files\Headway MT5 Terminal\terminal64.exe`).
   - Add your API keys (`GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, etc.).

---

## Phase 4: Keeping it Running 24/7 (Process Management)

If you simply run the generic python commands and close the RDP window, Windows might suspend the processes. We use **PM2** to keep them alive and automatically restart them if they crash.

1. Open a fresh PowerShell window (as Administrator) and install PM2 globally:
   ```powershell
   npm install pm2 -g
   ```
2. Create a startup script for the **Execution Server**. In your bot folder, create a file named `start_api.bat`:
   ```bat
   @echo off
   cd C:\Users\Administrator\Desktop\trading-bot
   call .venv\Scripts\activate
   uvicorn modules.execution_server:app --host 0.0.0.0 --port 8000
   ```
3. Create a startup script for the **Trading Bot**. Create a file named `start_bot.bat`:
   ```bat
   @echo off
   cd C:\Users\Administrator\Desktop\trading-bot
   call .venv\Scripts\activate
   python main.py
   ```
4. Start both using PM2:
   ```powershell
   pm2 start start_api.bat --name "ExecutionAPI"
   pm2 start start_bot.bat --name "TradingBot"
   ```
5. Ensure they restart if the VPS reboots:
   - Check PM2 documentation for Windows startup scripts, or simply place a shortcut to a `.bat` file running `pm2 resurrect` in the Windows Startup folder (`shell:startup`).

To view logs anytime:
```powershell
pm2 logs ExecutionAPI
pm2 logs TradingBot
```

---

## Phase 5: Exposing the API to the Internet securely

Right now, your API exists on `localhost:8000` on the VPS. To let external dashboards fetch data, you need to expose it. The easiest, most secure method without dealing with Windows Firewall/Router ports is **Cloudflare Tunnels**.

1. Create a free account at [Cloudflare](https://dash.cloudflare.com/) and add your domain name.
2. Go to **Zero Trust** -> **Networks** -> **Tunnels**.
3. Create a new tunnel (e.g., named "TradingBotAPI").
4. Under "Install and run a connector", choose **Windows** and copy the provided command. Run this command in PowerShell on your VPS.
5. In the Cloudflare UI, route public traffic:
   - **Public Hostname:** `api.yourdomain.com`
   - **Service Type:** `HTTP`
   - **Service URL:** `localhost:8000`

Boom! Your API is now securely accessible from anywhere in the world via `HTTPS://api.yourdomain.com/signals/current`, perfectly encrypted, and DDoS protected, all without opening a single port on your VPS.
