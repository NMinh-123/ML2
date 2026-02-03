import pandas as pd
import joblib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(BASE_DIR, '1_data', 'processed', 'FD001_cleandata.csv')

MODEL_PATH = os.path.join(BASE_DIR, '5_training', 'saved_models', 'ocsvm_fd001.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '5_training', 'saved_models', 'scaler_fd001.pkl')

OUTPUT_DIR = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'scored_data.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Không tìm thấy Model hoặc Scaler! Hãy chạy file '5_training/train_ocsvm.py' trước.")
    
    print("⏳ Đang tải model và dữ liệu...")
    ocsvm = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = [c for c in df.columns if c not in ['engine_id', 'cycle']]
    X_raw = df[feature_cols]

    X_scaled = scaler.transform(X_raw)

    print("Đang tính toán điểm bất thường (Anomaly Scores)...")
    scores = ocsvm.decision_function(X_scaled)
    preds = ocsvm.predict(X_scaled)
    results = pd.DataFrame()
    results['engine_id'] = df['engine_id']
    results['cycle'] = df['cycle']
    results['anomaly_score'] = scores
    results['prediction'] = preds
    results.to_csv(OUTPUT_FILE, index=False)
    
    print("Hoàn tất chấm điểm!")
    print(f"Kết quả đã lưu tại: {OUTPUT_FILE}")
    print("-" * 30)
    print("Mẫu kết quả:")
    print(results.head())

if __name__ == "__main__":
    main()