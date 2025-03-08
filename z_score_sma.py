from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    Exchange,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
    Symbol,
)
from cybotrade.permutation import Permutation
import talib  # 技術分析函式庫，用於計算技術指標
import numpy as np  # 數值計算函式庫
import asyncio  # 非同步程式設計
import logging  # 日誌記錄
import colorlog  # 彩色日誌輸出
import math  # 數學函式庫

# 定義運行模式：實時交易模式
RUNTIME_MODE = RuntimeMode.Live

# 自定義交易策略類，繼承自 BaseStrategy
class Strategy(BaseStrategy):
    # 策略相關參數
    symbol = [Symbol(base="BTC", quote="USDT")]  # 交易對
    quantity = 0.001  # 每次交易的數量
    hedge_mode = True  # 是否啟用對沖模式
    sma_length = 50  # 簡單移動平均線的長度
    z_score_threshold = 0.75  # Z-Score 的閾值

    # 設定參數的方法
    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        # 設定簡單移動平均線長度
        if identifier == "sma":
            self.sma_length = float(value)
        # 設定 Z-Score 閾值
        elif identifier == "z_score":
            self.z_score_threshold = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    # 將毫秒轉換為 datetime 對象
    def convert_ms_to_datetime(self, milliseconds):
        seconds = milliseconds / 1000.0  # 將毫秒轉換為秒
        return datetime.fromtimestamp(seconds)  # 返回 datetime 對象

    # 計算數組的平均值
    def get_mean(self, array):
        total = 0
        for i in range(0, len(array)):
            total += array[i]
        return total / len(array)

    # 計算數組的標準差
    def get_stddev(self, array):
        total = 0
        mean = self.get_mean(array)  # 計算平均值
        for i in range(0, len(array)):
            minus_mean = math.pow(array[i] - mean, 2)  # 計算每個數據與平均值的平方差
            total += minus_mean
        return math.sqrt(total / (len(array) - 1))  # 返回標準差

    # 初始化方法
    def __init__(self):
        # 設置日誌輸出格式
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}"))
        file_handler = TimedRotatingFileHandler(
            filename="z_score_sma.log", when="h", backupCount=10
        )  # 設置滾動日誌文件
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    # 當 K 線結束時觸發的事件
    async def on_candle_closed(self, strategy, topic, symbol):
        # 獲取對應交易對的 K 線數據
        candles = self.data_map[topic]
        # 獲取收盤價數據
        close = np.array(list(map(lambda c: float(c["close"]), candles)))
        # 獲取 K 線的開始時間
        start_time = np.array(list(map(lambda c: float(c["start_time"]), candles)))
        # 計算簡單移動平均線
        sma_forty = talib.SMA(close, self.sma_length)
        # 計算標準差
        std = self.get_stddev(close[-50:])
        # 計算 Z-Score
        z_score = (close[-1] - sma_forty[-1]) / std

        # 獲取當前持倉
        current_pos = await strategy.position(exchange=Exchange.BybitLinear, symbol=symbol)
        logging.info(
            f"close: {close[-1]}, sma: {sma_forty[-1]}, std: {std}, z_score: {z_score}, current_pos: {current_pos} at {self.convert_ms_to_datetime(start_time[-1])}"
        )

        # 如果 Z-Score 超過設定閾值，則開多倉
        if z_score > self.z_score_threshold:
            if current_pos.long.quantity == 0.0:  # 確保當前沒有多倉
                try:
                    await strategy.open(
                        exchange=Exchange.BybitLinear,
                        side=OrderSide.Buy,
                        quantity=self.quantity,
                        symbol=symbol,
                        limit=None,
                        take_profit=None,
                        stop_loss=None,
                        is_hedge_mode=self.hedge_mode,
                        is_post_only=False,
                    )
                except Exception as e:
                    logging.error(f"Failed to open long: {e}")
        # 如果 Z-Score 未達閾值，則平倉
        else:
            if current_pos.long.quantity != 0.0:  # 確保當前有多倉
                try:
                    await strategy.close(
                        exchange=Exchange.BybitLinear,
                        side=OrderSide.Buy,
                        quantity=self.quantity,
                        symbol=symbol,
                        is_hedge_mode=self.hedge_mode,
                    )
                except Exception as e:
                    logging.error(f"Failed to close entire position: {e}")

        # 紀錄新的持倉狀態
        new_pos = await strategy.position(exchange=Exchange.BybitLinear, symbol=symbol)
        logging.info(f"new_pos: {new_pos}")


# 配置運行時參數
config = RuntimeConfig(
    mode=RUNTIME_MODE,  # 運行模式
    datasource_topics=[],  # 數據源主題
    candle_topics=["candles-1d-BTC/USDT-bybit"],  # 訂閱的 K 線主題
    active_order_interval=1,  # 活躍訂單檢查間隔
    initial_capital=10000.0,  # 初始資金
    exchange_keys="./credentials.json",  # API 憑據文件路徑
    start_time=datetime(2022, 6, 11, 0, 0, 0, tzinfo=timezone.utc),  # 回測開始時間
    end_time=datetime(2024, 1, 5, 0, 0, 0, tzinfo=timezone.utc),  # 回測結束時間
    data_count=150,  # 加載的數據數量
    api_key="test",  # API Key (僅用於測試)
    api_secret="notest",  # API Secret (僅用於測試)
)

# 配置超參數
permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["sma"] = [50]  # 設定簡單移動平均線的長度
hyper_parameters["z_score"] = [0.75]  # 設定 Z-Score 閾值

# 定義策略運行入口
async def start():
    await permutation.run(hyper_parameters, Strategy)

# 啟動策略
asyncio.run(start())
