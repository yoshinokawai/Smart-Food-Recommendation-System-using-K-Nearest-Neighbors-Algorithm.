import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Thêm thư mục chứa src vào sys.path để import dễ dàng
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from recommender import SmartFoodRecommender
from evaluator import ModelEvaluator

def train_models():
    """
    Quy trình huấn luyện, chia tập dữ liệu Train/Test và đánh giá các mô hình.
    Lưu kết quả các mô hình (.pkl) và chỉ số đánh giá (.json).
    """
    print("==================================================")
    print(" BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH")
    print("==================================================")

    # 1. Tải và tiền xử lý dữ liệu
    loader = DataLoader()
    df_foods = loader.generate_dummy_data()
    preprocessor = DataPreprocessor()
    
    # Lấy toàn bộ ma trận đặc trưng
    X_full = preprocessor.get_feature_matrix(df_foods)
    
    # Thư mục lưu mô hình
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Train/Test Split & Phân loại (Classification)
    # Đặc trưng dự đoán (X): spicy, sweet, price_range, region_code
    # Nhãn mục tiêu (y): is_vegan (Dự đoán món ăn là chay hay mặn)
    features_cols = ['is_spicy', 'is_sweet', 'price_range', 'region_code']
    X_clf = df_foods[features_cols].values
    y_clf = df_foods['is_vegan'].values
    
    # Chia tập dữ liệu: 80% train, 20% test
    # Sử dụng stratify=y_clf để đảm bảo cân bằng tỷ lệ chay/mặn giữa 2 tập
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    print(f"[*] Kích thước tập Train: {len(X_train)} món ăn")
    print(f"[*] Kích thước tập Test: {len(X_test)} món ăn")
    
    # Huấn luyện Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)
    
    # Dự đoán trên tập Test
    y_pred = clf.predict(X_test)
    
    # Tính toán các chỉ số đánh giá phân loại
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI (RANDOM FOREST) ---")
    print(f"  + Độ chính xác (Accuracy): {accuracy:.4f}")
    print(f"  + Precision: {precision:.4f}")
    print(f"  + Recall: {recall:.4f}")
    print(f"  + F1-Score: {f1:.4f}")
    print("  + Confusion Matrix (Ma trận nhầm lẫn):")
    print(cm)
    
    # 3. Phân cụm (Clustering) KMeans
    evaluator = ModelEvaluator()
    optimal_k = evaluator.evaluate_clusters(X_full, max_k=5)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_full)
    best_silhouette = silhouette_score(X_full, cluster_labels)
    
    # 4. Huấn luyện hệ gợi ý KNN Recommender
    recommender = SmartFoodRecommender(n_neighbors=3)
    recommender.fit(df_foods, X_full)
    
    # 5. Lưu trữ các mô hình (Model Persistence)
    with open(os.path.join(models_dir, 'classifier_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(models_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(models_dir, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(recommender, f)
        
    print("\n[*] Đã lưu tất cả các mô hình (.pkl) thành công vào thư mục: models/")
    
    # 6. Lưu chỉ số đánh giá thành file JSON
    metrics = {
        'classification_metrics': {
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'accuracy': float(round(accuracy, 4)),
            'precision': float(round(precision, 4)),
            'recall': float(round(recall, 4)),
            'f1_score': float(round(f1, 4)),
            'confusion_matrix': cm.tolist()  # [[TN, FP], [FN, TP]]
        },
        'clustering_metrics': {
            'optimal_k': int(optimal_k),
            'silhouette_score': float(round(best_silhouette, 4))
        }
    }
    
    with open(os.path.join(models_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
        
    print("[*] Đã lưu các chỉ số đánh giá vào models/metrics.json")
    print("==================================================\n")

if __name__ == '__main__':
    train_models()
