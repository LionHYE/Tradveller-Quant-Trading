from datetime import datetime, timezone
from typing import Dict, List
from cybotrade.strategy import Strategy as BaseStrategy  # 引入基礎策略類
from cybotrade.runtime import StrategyTrader  # 引入策略執行器
from cybotrade.models import (
    RuntimeConfig,  # 運行時配置
    RuntimeMode,  # 運行模式
)
from cybotrade.permutation import Permutation  # 用於參數排列組合的工具
import numpy as np  # 數值計算函式庫
import asyncio  # 非同步程式設計
import logging  # 日誌記錄
import colorlog  # 彩色日誌輸出
import pandas as pd  # 用於數據處理與導出

# 自定義策略類，繼承自 BaseStrategy
class Strategy(BaseStrategy):
    datasource_data = []  # 用於存儲數據源數據
    candle_data = []  # 用於存儲 K 線數據
    start_time = datetime.utcnow()  # 記錄策略開始運行的時間

    def __init__(self):
        # 設置日誌輸出格式
        handler = colorlog.StreamHandler()  # 終端輸出日誌
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        # 配置日誌文件輸出
        file_handler = logging.FileHandler("cryptoquant-exchange-inflow-log-extraction.log")
        file_handler.setLevel(logging.INFO)  # 設置日誌等級為 INFO
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])  # 初始化策略類

    # 設置策略參數的方法
    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")  # 記錄參數設置的日誌
        if identifier == "sma":  # 設置移動平均線長度
            self.sma_length = int(value)
        elif identifier == "z_score":  # 設置 Z-Score 閾值
            self.z_score_threshold = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")  # 記錄錯誤的參數設置

    # 當 K 線結束時觸發的事件
    async def on_candle_closed(self, strategy, topic, symbol):
        # 獲取關聯交易對的最新 K 線數據
        logging.info("candle data {}".format(super().data_map[topic][-1]))
        # 將最新的 K 線數據存入 candle_data 列表
        self.candle_data.append(super().data_map[topic][-1])

    # 當數據源的數據到達時觸發的事件
    async def on_datasource_interval(self, strategy: StrategyTrader, topic: str, data_list):
        # 獲取最新的數據源數據
        logging.info("datasource data {}".format(super().data_map[topic][-1]))
        # 將數據存入 datasource_data 列表
        self.datasource_data.append(super().data_map[topic][-1])

    # 當回測完成時觸發的事件
    async def on_backtest_complete(self, strategy: StrategyTrader):
        # 將數據源數據轉換為 DataFrame（如需要）
        # df = pd.DataFrame(self.datasource_data)
        # 將數據保存為 CSV 文件（此處註釋掉）
        # df.to_csv("coinglass_openinterest_ohlc_binance_BTC_4h.csv")

        # 將 K 線數據轉換為 DataFrame 並保存為 CSV 文件
        df = pd.DataFrame(self.candle_data)
        df.to_csv("bybit_candle_btc_1d.csv")  # 將 Bybit 的 BTC/USDT 日線數據保存為 CSV 文件

        # 計算回測所用的總時間並打印
        time_taken = datetime.utcnow() - self.start_time
        print("Total time taken: ", time_taken)

# 配置運行時參數
config = RuntimeConfig(
    mode=RuntimeMode.Backtest,  # 設置為回測模式
    datasource_topics=[],  # 數據源主題（此處為空）
    candle_topics=["candles-1d-BTC/USDT-bybit"],  # 訂閱的 K 線主題（Bybit 的 BTC/USDT 日線）
    active_order_interval=1,  # 設置活躍訂單檢查間隔
    initial_capital=10_000.0,  # 初始資金
    exchange_keys="./asdfasd.json",  # API 憑據文件路徑
    start_time=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 回測開始時間
    end_time=datetime(2024, 4, 1, 5, 0, 0, tzinfo=timezone.utc),  # 回測結束時間
    data_count=100,  # 加載的數據數量
    api_key="test",  # 測試用 API Key
    api_secret="notest",  # 測試用 API Secret
)

# 配置超參數排列組合工具
permutation = Permutation(config)
hyper_parameters = {}  # 超參數配置（此處為空）

# 策略運行入口
async def start():
    await permutation.run(hyper_parameters, Strategy)  # 運行策略並應用超參數排列組合

# 啟動策略
asyncio.run(start())
