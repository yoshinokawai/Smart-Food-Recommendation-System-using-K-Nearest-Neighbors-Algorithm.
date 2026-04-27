from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

class SmartFoodRecommender:
    """
    Thuật toán cốt lõi: Gợi ý món ăn sử dụng K-Nearest Neighbors (KNN) 
    và độ đo Cosine Similarity.
    """
    def __init__(self, n_neighbors=3):
        # Số lượng món ăn gợi ý
        self.n_neighbors = n_neighbors
        # Khởi tạo thuật toán KNN với khoảng cách Cosine
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        self.feature_matrix = None
        self.df_foods = None

    def fit(self, df_foods, feature_matrix):
        """
        Huấn luyện mô hình với ma trận đặc trưng.
        """
        self.df_foods = df_foods
        self.feature_matrix = feature_matrix
        self.knn_model.fit(self.feature_matrix)

    def recommend(self, user_vector):
        """
        Tìm kiếm láng giềng gần nhất (các món ăn tương đồng nhất) 
        dựa trên vector sở thích của người dùng.
        """
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=self.n_neighbors)
        
        # Trích xuất thông tin các món ăn tương ứng
        recommended_foods = self.df_foods.iloc[indices[0]]
        return recommended_foods, distances[0]
