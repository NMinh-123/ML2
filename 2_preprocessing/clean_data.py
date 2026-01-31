import os
import pandas as pd
import numpy as np
cols = ['engine_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '1_data', 'raw', 'train_FD001.txt')
OUTPUT_PATH = os.path.join(BASE_DIR, '1_data', 'processed', 'FD001_cleandata.csv')

def main():
    print(f"1. Đọc dữ liệu từ: {DATA_PATH}")
    # Đọc file txt
    df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None, names=cols)

    # Lọc dữ liệu bình thường (30 chu kỳ đầu)
    N_NORMAL = 30
    normal_df = df[df['cycle'] <= N_NORMAL].copy()

    # Bỏ các cảm biến nhiễu (LIST CỐ ĐỊNH - KHÔNG DÙNG TỰ ĐỘNG)
    drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    
    # Chỉ giữ lại các cột feature cần thiết
    feature_cols = [c for c in df.columns if (c.startswith('op_') or c.startswith('sensor_')) and c not in drop_sensors]
    
    print(f"   -> Giữ lại {len(feature_cols)} đặc trưng (Features).")

    # Làm mượt (Rolling Mean) nhưng KHÔNG chuẩn hóa
    X_smooth = (
        normal_df.groupby('engine_id')[feature_cols]
        .rolling(window=5, min_periods=1).mean()
        .reset_index(drop=True)
    )

    # LƯU DỮ LIỆU THÔ (QUAN TRỌNG: Giá trị phải lớn, ví dụ 642.xx)
    df_final = pd.concat([normal_df[['engine_id', 'cycle']].reset_index(drop=True), X_smooth], axis=1)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Đã lưu file sạch (CHƯA SCALE) tại: {OUTPUT_PATH}")
    print("Hãy mở file này ra kiểm tra: Nếu thấy số > 500 là ĐÚNG. Thấy số < 5 là SAI.")

if __name__ == "__main__":
    main()