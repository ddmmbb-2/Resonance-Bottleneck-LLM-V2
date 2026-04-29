import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def plot_monitor():
    plt.ion() 
    # 只保留一個主圖（Loss），並共用 X 軸畫 LR
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1_lr = ax1.twinx()

    while True:
        try:
            # 檢查檔案是否存在
            if not os.path.exists("train_log_v18.csv"):
                print("等待 train_log_v18.csv 檔案建立中...")
                time.sleep(5)
                continue
            
            # 使用更強健的讀取方式
            df = pd.read_csv("train_log_v18.csv", engine='python', on_bad_lines='skip')
            
            # 🎯 修正：對齊 V18 輸出的欄位
            required_cols = ['step', 'loss', 'lr']
            if not all(col in df.columns for col in required_cols):
                print(f"數據格式尚未就緒，目前欄位: {list(df.columns)}")
                time.sleep(5)
                continue

            if len(df) < 5: 
                print(f"數據點不足 (目前 {len(df)} 點)，等待中...")
                time.sleep(5)
                continue

            # 清除畫布
            ax1.clear()
            ax1_lr.clear()

            window = min(len(df), 20)
            smooth_loss = df['loss'].rolling(window=window).mean()

            # 繪製 Loss (紅色系)
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss (Log Scale)', color='tab:red')
            ax1.plot(df['step'], df['loss'], color='tab:red', alpha=0.2, label='Raw Loss')
            ax1.plot(df['step'], smooth_loss, color='tab:red', linewidth=2, label='Smooth Loss')
            ax1.set_yscale('log')
            
            # 繪製 Learning Rate (灰色系)
            ax1_lr.set_ylabel('Learning Rate', color='gray')
            ax1_lr.plot(df['step'], df['lr'], color='gray', linestyle='--', label='LR')

            # 🎯 修正：更新標題為 V18.1
            plt.suptitle('🌊 D2-V18.1 Training Monitor', fontsize=16)
            fig.tight_layout()
            plt.pause(10)

        except Exception as e:
            print(f"讀取中遇到的問題: {e}")
            time.sleep(5)

if __name__ == "__main__":
    plot_monitor()