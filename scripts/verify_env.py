
import sys
import os

def verify_imports():
    modules_to_test = [
        "ccxt", "pandas", "numpy", "dotenv", "yaml", "aiohttp", 
        "websockets", "requests", "vectorbt", "plotly", "rich", 
        "telegram"
    ]
    
    local_modules = [
        "modules.ai_signal_engine", "modules.data_engine", 
        "modules.indicator_engine", "modules.risk_manager"
    ]
    
    print("--- Testing Core Dependencies ---")
    for mod in modules_to_test:
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError as e:
            print(f"[FAIL] {mod}: {e}")
            
    print("\n--- Testing Local Project Modules ---")
    for mod in local_modules:
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError as e:
            print(f"[FAIL] {mod}: {e}")

if __name__ == "__main__":
    verify_imports()
