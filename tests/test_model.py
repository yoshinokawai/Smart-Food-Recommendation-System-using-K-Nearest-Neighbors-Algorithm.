import os
import sys
import unittest
import pandas as pd
import numpy as np

# Thêm thư mục gốc của dự án vào sys.path để có thể import các module trong src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.recommender import SmartFoodRecommender
from src.evaluator import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier

class TestFoodRecommendationSystem(unittest.TestCase):
    def setUp(self):
        """
        Thiết lập môi trường test: Khởi tạo DataLoader và Preprocessor.
        """
        self.loader = DataLoader()
        self.df_foods = self.loader.generate_dummy_data()
        self.preprocessor = DataPreprocessor()
        self.feature_matrix = self.preprocessor.get_feature_matrix(self.df_foods)

    def test_data_loading(self):
        """
        Xác minh dữ liệu tải lên đúng cấu trúc yêu cầu.
        """
        self.assertIsNotNone(self.df_foods)
        self.assertTrue(len(self.df_foods) >= 15, "Dataset phải có ít nhất 15 món ăn.")
        required_columns = ['food_id', 'food_name', 'is_spicy', 'is_vegan', 'is_sweet', 'price_range', 'region_code']
        for col in required_columns:
            self.assertIn(col, self.df_foods.columns, f"Cột {col} bị thiếu trong dữ liệu.")

    def test_preprocessing(self):
        """
        Kiểm tra ma trận đặc trưng được trích xuất chính xác.
        """
        self.assertEqual(self.feature_matrix.shape[0], len(self.df_foods))
        self.assertEqual(self.feature_matrix.shape[1], 5, "Ma trận đặc trưng phải có đúng 5 thuộc tính.")

    def test_knn_recommender(self):
        """
        Kiểm tra mô hình KNN Recommender huấn luyện và đưa ra gợi ý hợp lệ.
        """
        recommender = SmartFoodRecommender(n_neighbors=3)
        recommender.fit(self.df_foods, self.feature_matrix)
        
        # Giả lập sở thích người dùng: [Không cay, Chay, Ngọt, Vừa phải, Quốc tế]
        user_pref = [0, 1, 1, 2, 0]
        user_vector = self.preprocessor.process_cold_start_user(user_pref)
        
        recommended_df, distances = recommender.recommend(user_vector)
        
        # Xác minh kết quả gợi ý
        self.assertEqual(len(recommended_df), 3, "KNN phải trả về đúng 3 gợi ý.")
        self.assertEqual(len(distances), 3, "KNN phải trả về đúng 3 giá trị khoảng cách.")
        # Khoảng cách Cosine phải nằm trong khoảng [0, 1]
        for d in distances:
            self.assertTrue(0 <= d <= 1, "Khoảng cách Cosine phải nằm trong đoạn [0, 1].")

    def test_kmeans_clustering(self):
        """
        Kiểm tra bộ đánh giá phân cụm K-Means hoạt động bình thường.
        """
        evaluator = ModelEvaluator()
        # Test phân cụm với tối đa 4 cụm
        optimal_k = evaluator.evaluate_clusters(self.feature_matrix, max_k=4)
        self.assertTrue(2 <= optimal_k <= 4, "Số lượng cụm tối ưu K phải nằm trong khoảng từ 2 đến 4.")

    def test_classifier(self):
        """
        Kiểm tra mô hình phân loại Random Forest hoạt động.
        """
        X = self.feature_matrix[:, [0, 2, 3, 4]] # spicy, sweet, price, region
        y = self.df_foods['is_vegan'].values
        
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        
        # Dự đoán món ăn mới
        test_food = np.array([[0, 0, 1, 1]]) # Không cay, Không ngọt, Giá rẻ, Miền Bắc
        prediction = clf.predict(test_food)
        self.assertIn(prediction[0], [0, 1], "Kết quả dự đoán chay/mặn phải là 0 hoặc 1.")

if __name__ == '__main__':
    unittest.main()
