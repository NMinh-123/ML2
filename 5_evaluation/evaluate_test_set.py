import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

TEST_FILE = os.path.join(BASE_DIR, '1_data', 'raw', 'test_FD001.txt')

MODEL_PATH = os.path.join(BASE_DIR, '5_training', 'saved_models', 'ocsvm_fd001.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '5_training', 'saved_models', 'scaler_fd001.pkl')

OUTPUT_FILE = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'test_results.csv')

def preprocess_test_data(df):
    cols = (['engine_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)])
    df.columns = cols
    
    drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    df = df.drop(columns=drop_sensors, errors='ignore')
    
    feature_cols = [c for c in df.columns if c not in ['engine_id', 'cycle']]
    
    df_rolled = (
        df.groupby('engine_id')[feature_cols]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index()
    )
    
    if 'level_1' in df_rolled.columns:
        df_rolled = df_rolled.drop(columns=['level_1'])
        
    df_rolled['cycle'] = df['cycle'].reset_index(drop=True)
    return df_rolled

# =========================================================
# 3. CHẠY ĐÁNH GIÁ
# =========================================================
def main():
    print("⏳ Đang tải dữ liệu và model...")
    if not os.path.exists(TEST_FILE):
        print(f"❌ Không tìm thấy file test tại: {TEST_FILE}")
        return

    # Sửa lỗi SyntaxWarning bằng r''
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', header=None)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Chưa có model. Hãy chạy train_ocsvm.py trước!")
        return

    ocsvm = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("🧹 Đang xử lý dữ liệu Test...")
    df_clean = preprocess_test_data(df_test)
    
    # Tách feature
    feature_cols = [c for c in df_clean.columns if c.startswith('op_') or c.startswith('sensor_')]
    X_test = df_clean[feature_cols]
    
    print(f"   -> Số lượng features Test: {X_test.shape[1]} (Kỳ vọng: 18)")

    # Scaling
    X_scaled = scaler.transform(X_test)

    # Dự báo
    print("🔍 Đang chấm điểm sức khỏe...")
    scores = ocsvm.decision_function(X_scaled)
    preds = ocsvm.predict(X_scaled)

    # Lưu kết quả
    results = df_clean[['engine_id', 'cycle']].copy()
    results['anomaly_score'] = scores
    results['prediction'] = preds
    
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Đã lưu kết quả Test tại: {OUTPUT_FILE}")

    # Vẽ biểu đồ mẫu Engine 1
    print("📊 Đang vẽ biểu đồ mẫu...")
    subset = results[results['engine_id'] == 1]
    plt.figure(figsize=(10, 5))
    plt.plot(subset['cycle'], subset['anomaly_score'], label='Test Score', color='purple')
    plt.axhline(0, color='red', linestyle='--', label='Critical Threshold')
    plt.title('Test Data Evaluation - Engine 1')
    plt.xlabel('Cycle')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'test_engine_1.png')
    plt.savefig(plot_path)
    print(f"   Đã lưu biểu đồ tại: {plot_path}")

if __name__ == "__main__":
    main()