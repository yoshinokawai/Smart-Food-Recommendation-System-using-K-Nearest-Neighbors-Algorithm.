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

app = Flask(__name__)

# Khởi tạo mô hình (Tải 1 lần khi server chạy)
loader = DataLoader()
df_foods = loader.generate_dummy_data()

preprocessor = DataPreprocessor()
feature_matrix = preprocessor.get_feature_matrix(df_foods)

# Tìm K tối ưu và Silhouette score để hiển thị lên UI
evaluator = ModelEvaluator()
optimal_k = evaluator.evaluate_clusters(feature_matrix, max_k=5)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(feature_matrix)
best_score = silhouette_score(feature_matrix, cluster_labels)

# Huấn luyện mô hình KNN
recommender = SmartFoodRecommender(n_neighbors=3)
recommender.fit(df_foods, feature_matrix)

@app.route('/')
def index():
    """Trang chủ hiển thị thông tin và chỉ số đánh giá."""
    return render_template(
        'index.html', 
        optimal_k=optimal_k, 
        silhouette_score=round(best_score, 4)
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
