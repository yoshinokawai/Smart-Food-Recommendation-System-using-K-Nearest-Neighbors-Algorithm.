import numpy as np

class DataPreprocessor:
    """
    Lớp xử lý dữ liệu (Preprocessing).
    Đảm nhiệm việc chuẩn hóa, trích xuất đặc trưng và xử lý bài toán Cold Start.
    """
    def __init__(self):
        pass

    def get_feature_matrix(self, df):
        """
        Trích xuất các cột đặc trưng để tạo thành Vector đặc trưng (Feature Matrix).
        Đầu ra này sẽ được đưa vào mô hình KNN để tính toán độ tương đồng.
        """
        features = df[['is_spicy', 'is_vegan', 'is_sweet', 'price_range', 'region_code']]
        return features.values

    def process_cold_start_user(self, user_preferences):
        """
        Giải quyết bài toán Cold Start: Chuyển đổi sở thích của người dùng mới 
        từ khảo sát (survey) thành vector đặc trưng.
        
        user_preferences: list [Cay, Chay, Ngọt, Giá, Vùng_miền]
        Trả về: Vector dạng numpy array (2D) dùng cho KNN.
        """
        user_vector = np.array(user_preferences).reshape(1, -1)
        return user_vector
