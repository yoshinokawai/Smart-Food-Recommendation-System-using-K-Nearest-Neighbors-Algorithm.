import sys
import os
# Fix lỗi in tiếng Việt trong console Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, render_template, request, redirect, url_for

# Cấu hình đường dẫn để có thể import các module từ thư mục 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.recommender import SmartFoodRecommender
from src.evaluator import ModelEvaluator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pickle
import json

app = Flask(__name__)

# Đảm bảo các mô hình đã được huấn luyện
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
knn_path = os.path.join(models_dir, 'knn_model.pkl')
metrics_path = os.path.join(models_dir, 'metrics.json')

if not os.path.exists(knn_path) or not os.path.exists(metrics_path):
    print("[*] Không tìm thấy file mô hình hoặc file chỉ số. Đang tự động huấn luyện...")
    try:
        from src.train import train_models
        train_models()
    except Exception as e:
        print(f"Lỗi khi tự động huấn luyện mô hình: {e}")

# Tải dữ liệu và mô hình đã lưu
loader = DataLoader()
df_foods = loader.generate_dummy_data()

preprocessor = DataPreprocessor()

# Tải mô hình KNN Recommender từ file
try:
    with open(knn_path, 'rb') as f:
        recommender = pickle.load(f)
except Exception as e:
    print(f"Lỗi khi tải mô hình KNN: {e}")
    recommender = SmartFoodRecommender(n_neighbors=3)
    feature_matrix = preprocessor.get_feature_matrix(df_foods)
    recommender.fit(df_foods, feature_matrix)

# Tải các chỉ số đánh giá từ file JSON
metrics = {}
try:
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
except Exception as e:
    print(f"Lỗi khi tải metrics: {e}")

@app.route('/')
def index():
    """Trang chủ hiển thị thông tin và chỉ số đánh giá."""
    foods_list = df_foods.to_dict(orient='records')
    return render_template(
        'index.html', 
        metrics=metrics,
        foods=foods_list
    )

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """Trang khảo sát cho bài toán Cold Start."""
    if request.method == 'POST':
        # Thu thập dữ liệu form
        is_spicy = int(request.form.get('is_spicy'))
        is_vegan = int(request.form.get('is_vegan'))
        is_sweet = int(request.form.get('is_sweet'))
        price_range = int(request.form.get('price_range'))
        region_code = int(request.form.get('region_code'))
        
        # Chuyển đổi thành vector cho người dùng mới
        user_preferences = [is_spicy, is_vegan, is_sweet, price_range, region_code]
        user_vector = preprocessor.process_cold_start_user(user_preferences)
        
        # Gọi thuật toán gợi ý KNN
        recommended_df, distances = recommender.recommend(user_vector)
        
        # Đóng gói dữ liệu kết quả để render ra UI
        results = []
        for idx, (index, row) in enumerate(recommended_df.iterrows()):
            # Khoảng cách cosine từ 0->1, càng nhỏ càng giống
            distance = distances[idx]
            match_percent = round((1 - distance) * 100, 1)
            
            results.append({
                'food_name': row['food_name'],
                'type': 'Món Chay' if row['is_vegan'] == 1 else 'Món Mặn',
                'distance': round(distance, 4),
                'match_percent': match_percent
            })
            
        return render_template('result.html', results=results)
        
    return render_template('survey.html')

if __name__ == '__main__':
    # Chạy server Flask trên port 5000
    app.run(debug=True, port=5000)
