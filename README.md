# Smart Food Recommendation System

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KNN%20%7C%20K--Means-orange.svg)

## 📌 Giới thiệu dự án

**Smart Food Recommendation System** là một hệ thống gợi ý món ăn thông minh được phát triển dưới dạng ứng dụng Web (sử dụng Flask). Dự án áp dụng các kỹ thuật Học máy (Machine Learning) để phân tích sở thích người dùng và đưa ra các món ăn phù hợp nhất. 

Điểm nổi bật của hệ thống là khả năng giải quyết bài toán **Cold Start** (Khởi động lạnh) thông qua một biểu mẫu khảo sát trực quan để thu thập dữ liệu sở thích ban đầu của người dùng mới, từ đó sử dụng **KNN (K-Nearest Neighbors)** và **Cosine Similarity** để tìm kiếm món ăn tương đồng nhất. Đồng thời, hệ thống sử dụng thuật toán **K-Means** và chỉ số **Silhouette Score** để đánh giá và phân nhóm chất lượng dữ liệu món ăn.

---

## 🚀 Các tính năng chính

1. **Gợi ý món ăn thông minh**: Trích xuất các món ăn tương đồng với sở thích của người dùng thông qua thuật toán KNN với độ đo khoảng cách Cosine (Cosine Similarity).
2. **Xử lý Cold Start (Khởi động lạnh)**: Cung cấp giao diện khảo sát (Survey) để lấy thông tin sở thích trực tiếp từ người dùng mới (độ cay, chay/mặn, độ ngọt, mức giá, vùng miền).
3. **Đánh giá mô hình tự động**: Tích hợp thuật toán K-Means phân cụm và tự động tìm ra số cụm $K$ tối ưu dựa vào điểm **Silhouette Score**.
4. **Giao diện Web trực quan**: Ứng dụng web dễ sử dụng với HTML, Bootstrap cho phép tương tác trực tiếp với mô hình AI thay vì chạy qua dòng lệnh terminal.

---

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ**: Python 3.x
- **Web Framework**: Flask, Jinja2 Templates
- **Machine Learning**: `scikit-learn` (KNeighbors, KMeans, Silhouette Score)
- **Xử lý dữ liệu**: `pandas`, `numpy`
- **Giao diện người dùng**: HTML5, CSS3, Bootstrap 5

---

## 📁 Cấu trúc thư mục

```text
food-recommendation-system/
├── app/
│   ├── main.py              # File khởi chạy chính của ứng dụng Flask
│   └── templates/           # Thư mục chứa giao diện HTML
│       ├── base.html        # Giao diện khung
│       ├── index.html       # Trang chủ & Kết quả đánh giá K-Means
│       ├── survey.html      # Trang khảo sát thông tin người dùng
│       └── result.html      # Trang hiển thị kết quả gợi ý món ăn
├── src/                     # Mã nguồn xử lý Machine Learning cốt lõi
│   ├── __init__.py          
│   ├── data_loader.py       # Tải và khởi tạo dữ liệu giả lập (Dummy Data)
│   ├── evaluator.py         # Đánh giá phân cụm (Silhouette Score)
│   ├── preprocessing.py     # Tiền xử lý dữ liệu và tạo ma trận đặc trưng
│   └── recommender.py       # Thuật toán gợi ý KNN
├── data/                    # Thư mục chứa bộ dữ liệu (CSV, JSON...)
├── models/                  # Thư mục lưu các mô hình đã huấn luyện (.pkl)
├── notebooks/               # Các file Jupyter Notebook để thử nghiệm thuật toán
├── tests/                   # Thư mục chứa các file kiểm thử tự động (Unit Test)
├── docs/                    # Tài liệu dự án
├── requirements.txt         # Danh sách các thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn sử dụng (File này)
```

---

## ⚙️ Hướng dẫn cài đặt và chạy ứng dụng

### Bước 1: Clone dự án hoặc tải mã nguồn
Mở terminal/command prompt và di chuyển đến thư mục bạn muốn lưu dự án.

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment) (Khuyến nghị)
```bash
# Trên Windows
python -m venv venv
venv\Scripts\activate

# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Huấn luyện mô hình (Optional)
Hệ thống sẽ tự động huấn luyện các mô hình khi khởi chạy Server lần đầu tiên. Tuy nhiên, nếu bạn muốn huấn luyện thủ công và đánh giá các mô hình một cách độc lập:
```bash
python src/train.py
```
*Lệnh này sẽ thực hiện chia tập dữ liệu thành Train/Test (80/20), huấn luyện mô hình phân loại Random Forest, K-Means và KNN, sau đó lưu kết quả vào thư mục `models/`.*

### Bước 5: Chạy kiểm thử tự động (Unit Test)
Để xác minh độ chính xác của mã nguồn và tính hợp lệ của mô hình:
```bash
python -m unittest tests/test_model.py
```

### Bước 6: Khởi chạy Server
Từ thư mục gốc của dự án `food-recommendation-system`, chạy lệnh:
```bash
python app/main.py
```

### Bước 7: Truy cập trên trình duyệt
Mở trình duyệt Web (Chrome, Edge, Firefox,...) và truy cập vào địa chỉ:
👉 **http://127.0.0.1:5000**

---

## 🧠 Hoạt động của thuật toán

1. **Data Loading**: `DataLoader` tự động tải dữ liệu của 40 món ăn từ file `data/foods.csv`.
2. **Preprocessing**: Dữ liệu thô được chuyển đổi thành ma trận vector đặc trưng.
3. **Train/Test Split & Classification**:
   - Dữ liệu được phân chia thành 80% để huấn luyện và 20% để kiểm thử.
   - Thuật toán `RandomForestClassifier` được dùng để học cách phân loại món ăn là Chay (`is_vegan=1`) hay Mặn (`is_vegan=0`).
   - Các chỉ số **Accuracy**, **Precision**, **Recall**, **F1-Score** và **Confusion Matrix** được tính toán để đánh giá độ chính xác trên tập Test.
4. **Clustering (K-Means)**: `ModelEvaluator` tìm ra số cụm $K$ tối ưu dựa trên điểm số **Silhouette Score** lớn nhất khi thử phân cụm dữ liệu món ăn.
5. **Recommendation (KNN)**: Khi người dùng thực hiện khảo sát sở thích (`/survey`), câu trả lời được chuyển đổi thành vector đặc trưng. Mô hình gợi ý `SmartFoodRecommender` (sử dụng thuật toán `NearestNeighbors` với hàm khoảng cách `cosine`) tìm kiếm Top 3 món ăn gần với sở thích của người dùng nhất và hiển thị lên giao diện web. Mức độ phù hợp được tính bằng công thức: $(1 - distance) \times 100$.

---

## 📝 Định hướng phát triển tương lai
- Kết nối với cơ sở dữ liệu thực (SQLite, MySQL hoặc MongoDB) thay vì dùng Dummy Data.
- Mở rộng tập dữ liệu hàng ngàn món ăn.
- Tích hợp thêm các hệ thống Gợi ý dựa trên Lọc Cộng Tác (Collaborative Filtering).
- Cung cấp RESTful API cho ứng dụng di động.