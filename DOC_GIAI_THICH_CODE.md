# HƯỚNG DẪN GIẢI THÍCH CODE & CÂU HỎI PHẢN BIỆN BẢO VỆ ĐỒ ÁN

Tài liệu này được biên soạn chi tiết nhằm giúp bạn nắm vững toàn bộ **logic dòng code** trong dự án và chuẩn bị sẵn sàng các **câu trả lời thông minh nhất** để trả lời trực tiếp cho Thầy/Cô khi bảo vệ môn học Học Máy (Machine Learning).

---

## PHẦN I: GIẢI THÍCH LOGIC CHI TIẾT TỪNG FILE CODE

### 1. File Dữ liệu: `data/foods.csv`
*   **Logic:** Chứa danh sách 40 món ăn thực tế.
*   **Các đặc trưng đặc biệt dùng cho Học máy:**
    *   `is_spicy` (0/1), `is_vegan` (0/1), `is_sweet` (0/1): Thuộc tính nhị phân.
    *   `price_range` (1: Rẻ, 2: Vừa, 3: Đắt): Biến phân loại thứ tự.
    *   `region_code` (1: Bắc, 2: Trung, 3: Nam, 0: Khác): Biến phân loại danh nghĩa.
*   *Lưu ý khi trả lời giáo viên:* "Dữ liệu được số hóa hoàn toàn về dạng số (numerical) để các thuật toán K-Means, KNN và Random Forest có thể tính toán khoảng cách và học các quy luật toán học."

---

### 2. File: `src/data_loader.py`
```python
class DataLoader:
    def __init__(self, data_path=None):
        # Xác định đường dẫn tuyệt đối đến file foods.csv
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(base_dir, 'data', 'foods.csv')
```
*   **Giải thích:** Tạo đường dẫn động bằng hàm `os.path` để dù giáo viên có chạy code ở bất kỳ thư mục nào trên máy tính của họ thì chương trình cũng không bị lỗi đường dẫn (file not found).
```python
    def generate_dummy_data(self):
        if os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                return df
```
*   **Giải thích:** Đọc file CSV bằng thư viện `pandas` đưa vào cấu trúc bảng `DataFrame`. Nếu file CSV bị xóa mất, hàm có sẵn khối dữ liệu dự phòng (dictionary) để chương trình vẫn chạy được bình thường (cơ chế dự phòng sự cố - fallback).

---

### 3. File: `src/preprocessing.py`
```python
    def get_feature_matrix(self, df):
        features = df[['is_spicy', 'is_vegan', 'is_sweet', 'price_range', 'region_code']]
        return features.values
```
*   **Giải thích:** Lọc lấy 5 cột đặc trưng ăn uống làm đầu vào cho ML, loại bỏ cột `food_id` và `food_name` (vì tên món ăn là dạng chữ, không có giá trị tính toán toán học). `.values` chuyển dữ liệu từ dạng bảng Pandas sang mảng ma trận NumPy 2 chiều để huấn luyện.
```python
    def process_cold_start_user(self, user_preferences):
        user_vector = np.array(user_preferences).reshape(1, -1)
        return user_vector
```
*   **Giải thích:** Sở thích của người dùng mới gửi lên từ form khảo sát là danh sách 1 chiều (1D array). Hàm `.reshape(1, -1)` biến đổi nó thành ma trận 2 chiều kích thước $1 \times 5$. 
*   *Lý do:* Thư viện `scikit-learn` quy định dữ liệu truyền vào dự đoán bắt buộc phải là ma trận 2 chiều (2D array).

---

### 4. File: `src/recommender.py`
```python
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
```
*   **Giải thích:** Khởi tạo thuật toán KNN ở chế độ không giám sát để tìm kiếm láng giềng.
    *   `n_neighbors=3`: Yêu cầu tìm ra đúng 3 món ăn gần nhất.
    *   `metric='cosine'`: Dùng khoảng cách Cosine thay vì Euclid để so sánh góc lệch của hai vector sở thích, đo lường độ tương đồng về mặt hành vi/gu ẩm thực tốt hơn.
```python
    def recommend(self, user_vector):
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=self.n_neighbors)
        recommended_foods = self.df_foods.iloc[indices[0]]
        return recommended_foods, distances[0]
```
*   **Giải thích:** Hàm `.kneighbors()` nhận vào vector người dùng, tính khoảng cách và trả về:
    *   `indices[0]`: Danh sách chỉ số dòng (index) của 3 món ăn phù hợp nhất. Từ đó dùng lệnh `.iloc` để trích xuất thông tin món ăn từ DataFrame gốc.
    *   `distances[0]`: Khoảng cách Cosine của 3 món ăn này (giá trị từ 0 đến 1, càng gần 0 càng giống).

---

### 5. File: `src/evaluator.py`
```python
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            score = silhouette_score(feature_matrix, cluster_labels)
```
*   **Giải thích:** Chạy thuật toán K-Means thử nghiệm từ 2 đến 5 cụm (`max_k=5`). Với mỗi số cụm $K$, gọi hàm `silhouette_score` để tính toán hệ số bóng đổ trung bình của toàn bộ dữ liệu. Điểm số này đo lường xem các cụm được phân chia có chặt chẽ và tách biệt rõ ràng hay không. Số cụm $K$ nào cho điểm số cao nhất sẽ được tự động chọn làm cấu trúc phân cụm tối ưu của thực đơn món ăn.

---

### 6. File: `src/train.py`
```python
    X_clf = df_foods[features_cols].values
    y_clf = df_foods['is_vegan'].values
```
*   **Giải thích:** Chuẩn bị dữ liệu cho mô hình phân loại Random Forest. Đặc trưng ($X$) gồm 4 cột (bỏ cột `is_vegan`), Nhãn mục tiêu ($y$) cần dự đoán là cột `is_vegan` (Chay/Mặn).
```python
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
```
*   **Giải thích:** Chia dữ liệu thành 80% để học (Train) và 20% để kiểm tra (Test).
    *   `stratify=y_clf`: Giữ nguyên tỷ lệ các lớp Chay/Mặn giữa tập Train và tập Test bằng đúng tỷ lệ trong tập dữ liệu gốc, tránh việc chia ngẫu nhiên làm mất cân bằng nhãn giữa các tập.
```python
    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)
```
*   **Giải thích:** Tạo mô hình Rừng ngẫu nhiên chứa 15 cây quyết định độc lập và huấn luyện trên tập Train bằng phương thức `.fit()`.
```python
    with open(os.path.join(models_dir, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(recommender, f)
```
*   **Giải thích:** Lưu trữ mô hình nhị phân đã huấn luyện thành file `.pkl`. Quá trình này gọi là **Model Persistence**. Nó giúp lưu lại trạng thái đã học của mô hình, khi Flask chạy chỉ cần nạp lại file này mà không phải tính toán huấn luyện lại từ đầu.

---

### 7. File: `app/main.py`
```python
if not os.path.exists(knn_path) or not os.path.exists(metrics_path):
    from src.train import train_models
    train_models()
```
*   **Giải thích:** Tự động kiểm tra file mô hình khi khởi động Web Server. Nếu file mô hình chưa có, nó sẽ tự động kích hoạt hàm huấn luyện để tạo mô hình ngay lập tức, ngăn ngừa lỗi ứng dụng khi mới tải về máy giáo viên.
```python
@app.route('/')
def index():
    foods_list = df_foods.to_dict(orient='records')
    return render_template('index.html', metrics=metrics, foods=foods_list)
```
*   **Giải thích:** Khi người dùng vào trang chủ, nạp kết quả đánh giá (accuracy, confusion matrix) từ file JSON và danh sách món ăn từ CSV truyền qua Template Engine `index.html` để render ra giao diện.

---

## PHẦN II: CÁC CÂU HỎI PHẢN BIỆN THƯỜNG GẶP CỦA THẦY/CÔ

### ❓ Câu 1: Bài toán "Cold Start" trong hệ gợi ý là gì? Dự án của em giải quyết nó như thế nào?
*   **Trả lời:** 
    *   *Định nghĩa:* Cold Start là hiện tượng hệ thống gợi ý không thể đưa ra gợi ý cho **người dùng mới** vì chưa có dữ liệu lịch sử hành vi (like, click, ratings).
    *   *Giải pháp trong đồ án:* Nhóm giải quyết bằng phương pháp **Lọc dựa trên nội dung (Content-based)** kết hợp trang **Khảo sát sở thích đầu vào (Survey)**. Khi người dùng mới trả lời 5 câu hỏi khảo sát, hệ thống chuyển câu trả lời thành vector đặc trưng rồi dùng thuật toán KNN so sánh với đặc trưng các món ăn để gợi ý ngay lập tức, bỏ qua giai đoạn lạnh.

### ❓ Câu 2: Tại sao em lại chọn độ đo khoảng cách Cosine thay vì khoảng cách Euclid cho hệ gợi ý KNN?
*   **Trả lời:** 
    *   Khoảng cách Euclid đo khoảng cách tuyệt đối về độ lớn giữa hai điểm. 
    *   Tuy nhiên, trong hệ gợi ý thuộc tính món ăn, các đặc trưng hầu hết là nhị phân (0 hoặc 1) và biểu thị khẩu vị. Độ đo **Cosine** đo góc lệch giữa 2 vector, giúp xác định hai thực thể có cùng xu hướng/hướng gu ẩm thực hay không, không phụ thuộc vào độ lớn tuyệt đối của giá trị. Vì vậy, Cosine Similarity mang lại độ chính xác gợi ý tốt hơn cho bài toán so sánh gu ẩm thực.

### ❓ Câu 3: "Stratify" trong hàm `train_test_split` có vai trò gì? Nếu không dùng thì sao?
*   **Trả lời:** 
    *   `stratify` là kỹ thuật phân lớp phân tầng. Nó đảm bảo tỷ lệ phân bổ các lớp mục tiêu (món Chay và món Mặn) ở hai tập dữ liệu con (Train và Test) luôn tương đương với tỷ lệ trong tập dữ liệu gốc ban đầu.
    *   *Nếu không dùng:* Với tập dữ liệu nhỏ (40 món ăn), việc phân chia ngẫu nhiên thuần túy rất dễ xảy ra tình trạng "lệch nhãn" (ví dụ tập Test vô tình chỉ có toàn món Mặn mà không có món Chay nào). Điều này khiến kết quả đánh giá mô hình bị sai lệch và không chính xác.

### ❓ Câu 4: Rừng ngẫu nhiên (Random Forest) hoạt động như thế nào? Tại sao nó lại tốt hơn một Cây quyết định (Decision Tree)?
*   **Trả lời:** 
    *   *Cách hoạt động:* Random Forest tạo ra nhiều cây quyết định hoạt động độc lập bằng cách lấy mẫu ngẫu nhiên dữ liệu (Bootstrap) và ngẫu nhiên thuộc tính (Feature Selection). Kết quả cuối cùng là kết quả bỏ phiếu số đông của tất cả các cây.
    *   *Tại sao tốt hơn:* Một cây quyết định đơn lẻ rất dễ gặp hiện tượng **quá khớp (Overfitting)** khi cố học quá chi tiết dữ liệu huấn luyện, dẫn đến dự đoán kém trên dữ liệu mới. Rừng ngẫu nhiên sử dụng nhiều cây và cơ chế bỏ phiếu giúp triệt tiêu phương sai sai số, giúp mô hình hoạt động ổn định và chính xác hơn trên tập Test.

### ❓ Câu 5: Em hãy giải thích ý nghĩa của chỉ số "Silhouette Score" trong thuật toán K-Means?
*   **Trả lời:** 
    *   Silhouette Score đo lường độ chất lượng của phân cụm. Nó tính toán dựa trên khoảng cách trung bình của một điểm dữ liệu đến các điểm khác trong cùng cụm (Cohesion - đo độ chặt chẽ cụm) so với khoảng cách đến cụm láng giềng gần nhất (Separation - đo độ tách biệt cụm).
    *   Giá trị nằm trong khoảng $[-1, 1]$. Giá trị càng gần 1 chứng tỏ các cụm được phân chia rất tốt, các điểm nằm khít nhau trong cụm và cách xa cụm khác. Hệ thống chạy K-Means từ 2 đến 5 cụm và tự động chọn số cụm $K$ có điểm Silhouette cao nhất để đảm bảo phân tách món ăn tự nhiên tốt nhất.

### ❓ Câu 6: Tại sao em lại cần lưu mô hình ra file đuôi `.pkl` (Pickle)?
*   **Trả lời:** 
    *   Huấn luyện mô hình học máy (như tìm tâm cụm KMeans hoặc dựng 15 cây quyết định Random Forest) là tiến trình tốn tài nguyên tính toán. 
    *   Việc lưu mô hình thành file nhị phân `.pkl` (tuần tự hóa mô hình) giúp hệ thống chỉ cần huấn luyện một lần rồi đóng gói lại. Khi chạy ứng dụng Web Flask, ta chỉ việc load file `.pkl` này lên trong vài phần nghìn giây là có thể sử dụng được ngay để dự đoán, giúp tối ưu hóa hiệu năng Web Server.
