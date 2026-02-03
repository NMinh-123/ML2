import pandas as pd
import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, '1_data', 'processed', 'FD001_cleandata.csv')
MODEL_DIR = os.path.join(BASE_DIR, '5_training', 'saved_models')
EXP_DIR = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline')

# Đảm bảo thư mục tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

def main():
    print("2. Đang huấn luyện mô hình...")
    df = pd.read_csv(INPUT_PATH)
    
    feature_cols = [c for c in df.columns if c.startswith('op_') or c.startswith('sensor_')]
    X_train = df[feature_cols].values

    # --- QUAN TRỌNG: CHUẨN HÓA TẠI ĐÂY ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # TRAIN MODEL
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    ocsvm.fit(X_scaled)

    # LƯU MODEL & SCALER
    joblib.dump(ocsvm, os.path.join(MODEL_DIR, 'ocsvm_fd001.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_fd001.pkl'))
    
    # LƯU KẾT QUẢ TRAIN (Để tính ngưỡng Threshold sau này)
    scores = ocsvm.decision_function(X_scaled)
    result_df = df[['engine_id', 'cycle']].copy()
    result_df['anomaly_score'] = scores
    
    # Lưu 2 file (1 file kết quả, 1 file cho script threshold dùng)
    result_df.to_csv(os.path.join(EXP_DIR, 'FD001_ocsvm_result.csv'), index=False)
    result_df.to_csv(os.path.join(EXP_DIR, 'scored_data.csv'), index=False)
    
    print("Đã huấn luyện xong!")
    print(f"   - Model lưu tại: {MODEL_DIR}")
    print(f"   - Dữ liệu train score lưu tại: {os.path.join(EXP_DIR, 'scored_data.csv')}")

if __name__ == "__main__":
    main()