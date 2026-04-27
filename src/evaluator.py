from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Lớp đánh giá mô hình bằng các chỉ số thống kê (như Silhouette Score).
    """
    def __init__(self):
        pass

    def evaluate_clusters(self, feature_matrix, max_k=5):
        """
        Tính toán chỉ số Silhouette Score để tìm số lượng cụm (K) tối ưu 
        cho tập dữ liệu món ăn. (Dùng thuật toán KMeans kết hợp).
        """
        best_k = 2
        best_score = -1
        
        print("--- ĐÁNH GIÁ CHẤT LƯỢNG PHÂN CỤM (SILHOUETTE SCORE) ---")
        for k in range(2, max_k + 1):
            if k >= len(feature_matrix):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Tính điểm Silhouette
            score = silhouette_score(feature_matrix, cluster_labels)
            print(f"  + Số cụm K = {k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                
        print(f"-> Gợi ý số cụm tối ưu: K = {best_k} (Điểm Silhouette cao nhất: {best_score:.4f})\n")
        return best_k
