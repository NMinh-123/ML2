import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

INPUT_FILE = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'scored_data.csv')

CONFIG_DIR = os.path.join(BASE_DIR, 'config')
os.makedirs(CONFIG_DIR, exist_ok=True)
OUTPUT_CONFIG_FILE = os.path.join(CONFIG_DIR, 'threshold_config.yaml')

N_NORMAL = 30 
def calculate_thresholds():
    if not os.path.exists(INPUT_FILE):
        print(f" Không tìm thấy file: {INPUT_FILE}. Hãy chạy Bước 6 trước.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📖 Đã tải {len(df)} dòng dữ liệu điểm số.")
    normal_scores = df[df['cycle'] <= N_NORMAL]['anomaly_score'].values
    
    print(f"📊 Phân tích phân phối trên {len(normal_scores)} mẫu bình thường:")
    mean_score = np.mean(normal_scores)
    std_score = np.std(normal_scores)
    print(f"   - Mean (Trung bình): {mean_score:.4f}")
    print(f"   - Std (Độ lệch chuẩn): {std_score:.4f}")
    print(f"   - Min: {np.min(normal_scores):.4f}")

    pct_1 = np.percentile(normal_scores, 1)   
    pct_5 = np.percentile(normal_scores, 5)   
    th_3sigma = mean_score - 3 * std_score
    
    thresholds = {
        "warning_level": float(pct_5),       
        "critical_level": float(0.0),        
        "extreme_level": float(th_3sigma)    
    }

    with open(OUTPUT_CONFIG_FILE, 'w') as file:
        yaml.dump(thresholds, file)

    print("\n🎯 ĐÃ XÁC ĐỊNH & LƯU NGƯỠNG CẢNH BÁO:")
    print(f"   WARNING (Vàng)   : {thresholds['warning_level']:.4f} (Dưới mức này -> Cảnh báo)")
    print(f"   CRITICAL (Đỏ)    : {thresholds['critical_level']:.4f} (Dưới mức này -> Báo động)")
    print(f"   EXTREME (Tím)    : {thresholds['extreme_level']:.4f} (Dưới mức này -> Hư hỏng nặng)")
    print(f"\nĐã lưu cấu hình tại: {OUTPUT_CONFIG_FILE}")

    plot_distribution(normal_scores, thresholds)

def plot_distribution(scores, thresholds):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='green', alpha=0.7, label='Normal Scores (Cycle <= 30)')
    
    plt.axvline(thresholds['warning_level'], color='orange', linestyle='--', linewidth=2, label='Warning Threshold')
    plt.axvline(thresholds['critical_level'], color='red', linestyle='-', linewidth=2, label='Critical Threshold (0)')
    
    plt.title('Phân phối điểm số của trạng thái Bình thường')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'threshold_dist.png')
    plt.savefig(plot_path)
    print(f"Biểu đồ phân phối đã lưu tại: {plot_path}")

if __name__ == "__main__":
    calculate_thresholds()
