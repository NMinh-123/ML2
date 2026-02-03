import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
RUL_FILE = os.path.join(BASE_DIR, '1_data', 'raw', 'RUL_FD001.txt')
TEST_RESULT_FILE = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'test_results.csv')
OUTPUT_IMG = os.path.join(BASE_DIR, 'experiments', 'exp_01_baseline', 'rul_vs_score.png')
def evaluate_rul():
    if not os.path.exists(RUL_FILE):
        print(f"Dont't find RUL file on: {RUL_FILE}")
        print("Please add file RUL_FD001.txt in folder '1_data/raw/'")
        return
    if not os.path.exists(TEST_RESULT_FILE):
        print("Don't find test file. Please run 'evaluate_test_set.py'.")
        return
    df_rul = pd.read_csv(RUL_FILE, header=None, names=['true_rul'])
    df_rul['engine_id'] = df_rul.index + 1
    df_results = pd.read_csv(TEST_RESULT_FILE)
    last_cycle_scores = df_results.groupby('engine_id').last().reset_index()
    merged = pd.merge(last_cycle_scores, df_rul, on='engine_id')
    print(merged[['engine_id', 'cycle', 'anomaly_score', 'true_rul']].head())
    print("Đang vẽ biểu đồ tương quan...")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged, x='true_rul', y='anomaly_score', hue='prediction', palette={1: 'green', -1: 'red'}, alpha=0.7)
    
    plt.axhline(0, color='red', linestyle='--', label='Critical Threshold (0)')
    
    plt.title('Mối quan hệ giữa RUL Thực tế và Điểm Bất thường (Anomaly Score)', fontsize=14)
    plt.xlabel('True RUL (Số chu kỳ còn lại thực tế)', fontsize=12)
    plt.ylabel('Last Anomaly Score (Điểm sức khỏe)', fontsize=12)
    plt.legend(title='Model Prediction')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.text(140, 0.1, 'Healthy Zone\n(RUL cao -> Score cao)', fontsize=10, color='green')
    plt.text(10, -0.2, 'Danger Zone\n(RUL thấp -> Score thấp)', fontsize=10, color='red')

    plt.savefig(OUTPUT_IMG)
    print(f"Biểu đồ đã lưu tại: {OUTPUT_IMG}")
    dying_engines = merged[merged['true_rul'] < 20]
    detected_engines = dying_engines[dying_engines['prediction'] == -1]
    
    if len(dying_engines) > 0:
        accuracy = len(detected_engines) / len(dying_engines) * 100
        print("\n--- ĐÁNH GIÁ KHẢ NĂNG CẢNH BÁO ---")
        print(f"Tổng số động cơ sắp hỏng (RUL < 20): {len(dying_engines)}")
        print(f"Số động cơ Model phát hiện được (Score < 0): {len(detected_engines)}")
        print(f"Độ nhạy (Sensitivity/Recall): {accuracy:.2f}%")
    else:
        print("\n(Không có động cơ nào sắp hỏng trong tập test để đánh giá độ nhạy)")

if __name__ == "__main__":
    evaluate_rul()