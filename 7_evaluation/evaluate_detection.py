import pandas as pd
import numpy as np
import os
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'scored_data.csv')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'threshold_config.yaml')
REPORT_PATH = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'evaluation_report.txt')

N_NORMAL_PERIOD = 30  
def evaluate_model():
    if not os.path.exists(DATA_PATH):
        print("Không tìm thấy dữ liệu score.")
        return

    df = pd.read_csv(DATA_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        thresholds = yaml.safe_load(f)
    critical_th = thresholds['critical_level']
    warning_th = thresholds['warning_level']
    normal_data = df[df['cycle'] <= N_NORMAL_PERIOD]
    total_normal_points = len(normal_data)
    false_alarms = len(normal_data[normal_data['anomaly_score'] < warning_th])
    
    far = false_alarms / total_normal_points if total_normal_points > 0 else 0

    lead_times = []
    
    for eid in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == eid].sort_values('cycle')
        max_cycle = engine_df['cycle'].max() 
        detected_points = engine_df[engine_df['anomaly_score'] < warning_th]
        
        if not detected_points.empty:
            first_detection_cycle = detected_points['cycle'].min()
            lead_time = max_cycle - first_detection_cycle
            lead_times.append(lead_time)
        else:
            lead_times.append(0)

    avg_lead_time = np.mean(lead_times)
    
    report = f"""
    ==================================================
    BÁO CÁO ĐÁNH GIÁ MÔ HÌNH OCSVM (FD001)
    ==================================================
    1. Cấu hình ngưỡng:
       - Warning Level : {warning_th:.4f}
       - Critical Level: {critical_th:.4f}

    2. Độ tin cậy (Reliability):
       - False Alarm Rate (FAR) trên 30 chu kỳ đầu: {far*100:.2f}%
       - (Mục tiêu: FAR < 5%)

    3. Khả năng dự báo sớm (Early Detection):
       - Trung bình Lead Time: {avg_lead_time:.1f} chu kỳ
       - Nghĩa là: Hệ thống cảnh báo trước khi hỏng trung bình {avg_lead_time:.1f} chu kỳ.
    
    4. Tổng kết:
       - Số động cơ được đánh giá: {len(df['engine_id'].unique())}
    ==================================================
    """
    
    print(report)
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"Báo cáo đã lưu tại: {REPORT_PATH}")

if __name__ == "__main__":
    evaluate_model()
