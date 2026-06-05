#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V24 訓練監控腳本（常駐版，只讀取最新 N 筆）
用法：python monitor.py --csv v24_samba_latent_log.csv [--recent 100]
啟動後每 60 秒自動更新一次，按 Ctrl+C 結束。
"""
import csv
import ast
import argparse
import numpy as np
import time
import os
from io import StringIO
from datetime import datetime

def tail(file_path, n):
    """讀取檔案最後 n 行（效率高，不掃描整個檔案），回傳字串"""
    with open(file_path, 'rb') as f:
        f.seek(0, 2)          # 移到檔案結尾
        file_size = f.tell()
        block_size = 1024
        data = b''
        lines_found = 0
        # 從尾端往前讀，直到收集到至少 n+1 個換行（確保 n 行）
        while file_size > 0 and lines_found <= n:
            read_size = min(block_size, file_size)
            file_size -= read_size
            f.seek(file_size)
            block = f.read(read_size)
            data = block + data
            lines_found = data.count(b'\n')
        # 分割並取最後 n 行（去除檔案結尾可能的空行）
        lines = data.split(b'\n')
        if lines and lines[-1] == b'':
            lines = lines[:-1]
        lines = lines[-n:]
        return b'\n'.join(lines).decode('utf-8', errors='replace')

def parse_list_str(s):
    """解析 Diffs 或 Halts 的字符串列表，如 '[0.21,0.24,0.26]'"""
    try:
        return ast.literal_eval(s)
    except:
        return []

def analyze(csv_path, recent=100):
    """只載入最後 recent 筆資料進行診斷"""
    if not os.path.exists(csv_path):
        print(f"⏳ 等待 CSV 檔案產生：{csv_path}")
        return False

    # ---------- 高效讀取：只拿 header + 最後 recent 行 ----------
    try:
        # 1. 讀取第一行（欄位名稱）
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            header = next(csv.reader([header_line]))

        # 2. 讀取最後 recent 行資料
        recent_data_str = tail(csv_path, recent)
        if not recent_data_str:
            print("⏳ 目前尚無訓練紀錄，等待中...")
            return False

        # 3. 組合成完整 CSV 文字，用 StringIO 解析
        csv_text = header_line + '\n' + recent_data_str
        reader = csv.DictReader(StringIO(csv_text))
        rows = []
        for row in reader:
            if not row.get('Step'):
                continue
            row['Step'] = int(row['Step'])
            row['Final_CE'] = float(row['Final_CE'])
            row['Align_Loss'] = float(row['Align_Loss'])
            row['Halt_Loss'] = float(row['Halt_Loss'])
            row['LR'] = float(row['LR'])
            row['Diffs'] = parse_list_str(row['Diffs'])
            row['Halts'] = parse_list_str(row['Halts'])
            rows.append(row)
    except Exception as e:
        print(f"❌ 讀取或解析 CSV 出錯：{e}")
        return False

    total_loaded = len(rows)
    if total_loaded == 0:
        print("⏳ 目前尚無訓練紀錄，等待中...")
        return False

    # 分析對象即為這些 rows（已經是最近 recent 筆）
    recent_rows = rows
    analyze_steps = len(recent_rows)
    steps = [r['Step'] for r in recent_rows]
    current_step = steps[-1]

    print(f"\n{'='*60}")
    print(f"📊 V24 訓練監控報告")
    print(f"   更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   本次分析最近 {analyze_steps} 筆紀錄 (Step {steps[0]} ~ {steps[-1]})")

    # 基本統計
    ces = [r['Final_CE'] for r in recent_rows]
    aligns = [r['Align_Loss'] for r in recent_rows]
    halts_loss = [r['Halt_Loss'] for r in recent_rows]
    lr = recent_rows[-1]['LR']

    # 每個推理核心的 Diffs 與 Halts（假設最多 3 個核心）
    core_diffs = [[] for _ in range(3)]
    core_halts = [[] for _ in range(3)]
    for r in recent_rows:
        for i in range(3):
            if i < len(r['Diffs']):
                core_diffs[i].append(r['Diffs'][i])
            if i < len(r['Halts']):
                core_halts[i].append(r['Halts'][i])

    avg_align = np.mean(aligns) if aligns else 0.0
    avg_halts_last = np.mean(core_halts[2]) if core_halts[2] else 0.0
    avg_halts_mid = np.mean(core_halts[1]) if core_halts[1] else 0.0
    avg_halts_first = np.mean(core_halts[0]) if core_halts[0] else 0.0
    avg_diffs = [np.mean(core_diffs[i]) if core_diffs[i] else 0.0 for i in range(3)]
    avg_ce = np.mean(ces)
    min_ce = np.min(ces)
    max_ce = np.max(ces)

    # CE 趨勢（分前後兩半）
    half = analyze_steps // 2
    if half >= 1:
        avg_ce_recent = np.mean(ces[-half:])
        avg_ce_older = np.mean(ces[:half])
        ce_improvement = avg_ce_older - avg_ce_recent
    else:
        avg_ce_recent = avg_ce_older = avg_ce
        ce_improvement = 0.0

    print(f"   當前學習率：{lr:.2e}")
    print(f"   CE 平均/最佳/最差：{avg_ce:.4f} / {min_ce:.4f} / {max_ce:.4f}")
    if half >= 1:
        trend_sign = "改善" if ce_improvement > 0 else "惡化"
        print(f"   CE 近期趨勢：{avg_ce_older:.4f} → {avg_ce_recent:.4f} ({trend_sign} {abs(ce_improvement):.4f})")
    print(f"   對齊損失 (Align)：{avg_align:.4f}（理想區間 0.05~0.3）")
    print(f"   Halt 機率（核心1/2/3）：{avg_halts_first:.3f} / {avg_halts_mid:.3f} / {avg_halts_last:.3f}")
    print(f"   平均 Diff（核心1/2/3）：{avg_diffs[0]:.3f} / {avg_diffs[1]:.3f} / {avg_diffs[2]:.3f}")

    # 診斷與建議（沿用原有邏輯，只根據 analyze_steps 調整）
    warnings = []
    suggestions = []

    # ---------- Align 檢查 ----------
    if current_step < 500:
        if avg_align > 0.5:
            warnings.append("⚡ 對齊損失偏高（>0.5），潛在步驟差異過大，可能造成不穩定。")
            suggestions.append("→ 降低對齊損失權重（目前 0.1）或檢查 Smooth L1 是否適合當前階段。")
        elif avg_align < 0.01:
            warnings.append("⚡ 對齊損失過低（<0.01），步驟幾乎無變化，可能對齊權重過低或模型尚未開始思考。")
            suggestions.append("→ 可稍微提高對齊權重至 0.15~0.2，但早期屬正常現象，暫無需動作。")
    else:
        if avg_align > 0.3:
            warnings.append("⚠️ 對齊損失偏高（>0.3），潛在步驟發散，可能導致最終輸出不穩定。")
            suggestions.append("→ 調降對齊權重至 0.05~0.1，或增加 diff 正則係數。")
        elif avg_align < 0.05:
            warnings.append("⚠️ 對齊損失過低（<0.05），步驟之間幾無區別，模型可能未充分利用思考步數。")
            suggestions.append("→ 提高對齊權重（0.15~0.2），或檢查步驟嵌入是否正常學習。")

    # ---------- Halt 檢查 ----------
    if avg_halts_last < 0.6:
        warnings.append(f"🔴 最後推理核心的 Halt 機率過低（{avg_halts_last:.3f}），應接近 0.7~0.9。")
        suggestions.append("→ 增加 Halt 損失權重（目前 0.05）或調整推理步數（think_steps）。")
    if avg_halts_first > 0.6:
        warnings.append(f"🟠 第一個推理核心的 Halt 機率偏高（{avg_halts_first:.3f}），可能太早停止思考。")
        suggestions.append("→ 加大 Halt 損失對早期步驟的懲罰，或調低 exit threshold 的初始偏置。")
    if avg_halts_mid > 0.6:
        warnings.append(f"🟠 第二個推理核心的 Halt 機率偏高（{avg_halts_mid:.3f}），可能中間步驟提早退出。")
        suggestions.append("→ 同上，檢查 Halt 損失設計。")

    # ---------- Diff 檢查 ----------
    for i, d in enumerate(avg_diffs):
        if d > 1.0:
            warnings.append(f"📈 核心 {i+1} 的平均 Diff 過大（{d:.3f}），潛在狀態更新劇烈，可能出現梯度爆炸。")
            suggestions.append("→ 檢查梯度裁剪（目前 max_norm=0.5），並考慮降低學習率或增加正則。")
        if d < 0.05 and current_step > 1000:
            warnings.append(f"📉 核心 {i+1} 的平均 Diff 過小（{d:.3f}），思考過程幾乎停滯，可能陷入局部最優。")
            suggestions.append("→ 略微增加自適應噪聲強度，或檢查路由閘門是否飽和。")

    # ---------- CE 趨勢檢查 ----------
    if half >= 1:
        if ce_improvement < -0.02 and current_step > 500:
            warnings.append(f"📉 最近 CE 平均上升 { -ce_improvement:.4f}，訓練可能不穩定或過擬合。")
            suggestions.append("→ 嘗試降低學習率，檢查資料是否重複，或增加 dropout。")
        elif ce_improvement < 0.002 and current_step > 2000:
            warnings.append("⏸️ CE 改善停滯（<0.002），可能已接近瓶頸。")
            suggestions.append("→ 考慮學習率衰減，或調整模型容量。")

    # ---------- 學習率檢查 ----------
    if lr < 5e-5 and current_step > 10000:
        suggestions.append("→ 學習率已很低，若 CE 不再下降，可提前停止訓練。")

    if warnings:
        print("\n🔍 警告：")
        for w in warnings:
            print("  " + w)
    else:
        print("\n✅ 無明顯異常，訓練狀態健康。")

    if suggestions:
        print("💡 建議處理方式：")
        for s in suggestions:
            print("  " + s)
    else:
        print("💡 目前無需調整。")

    print(f"{'='*60}\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="V24 訓練監控工具（常駐模式，高效讀取最新 N 筆）")
    parser.add_argument("--csv", type=str, default="v24_samba_latent_log.csv", help="CSV 日誌路徑")
    parser.add_argument("--recent", type=int, default=100, help="分析的最近筆數（預設 100）")
    parser.add_argument("--interval", type=int, default=60, help="更新間隔（秒，預設 60）")
    args = parser.parse_args()

    print(f"🚀 V24 監控啟動，每 {args.interval} 秒刷新一次，只讀取最新 {args.recent} 筆，按 Ctrl+C 終止。")
    try:
        while True:
            analyze(args.csv, args.recent)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n👋 監控已手動終止。")

if __name__ == "__main__":
    main()