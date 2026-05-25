import os
import pandas as pd

class DataLoader:
    """
    Lớp tải và quản lý dữ liệu cho hệ thống.
    """
    def __init__(self, data_path=None):
        if data_path is None:
            # Tìm đường dẫn tuyệt đối đến data/foods.csv
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_path = os.path.join(base_dir, 'data', 'foods.csv')
        else:
            self.data_path = data_path

    def generate_dummy_data(self):
        """
        Tải dữ liệu từ file CSV. Nếu không tồn tại, trả về dữ liệu giả lập.
        """
        if os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                return df
            except Exception as e:
                print(f"Lỗi khi đọc file CSV: {e}. Đang chuyển sang dùng dữ liệu mặc định.")

        data = {
            'food_id': range(1, 16),
            'food_name': ['Phở bò', 'Bún chả', 'Salad chay', 'Canh chua', 'Chè trôi nước', 
                          'Gà rán', 'Lẩu thái cay', 'Đậu hũ tứ xuyên', 'Bánh xèo', 'Gỏi cuốn chay',
                          'Trà sữa trân châu', 'Nước mía', 'Cà phê đen đá', 'Sinh tố bơ', 'Trà đào cam sả'],
            'is_spicy': [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            'is_vegan': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            'is_sweet': [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
            'price_range': [2, 2, 1, 2, 1, 3, 3, 2, 2, 1, 2, 1, 1, 2, 2], 
            'region_code': [1, 1, 0, 3, 1, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0]  
        }
        df = pd.DataFrame(data)
        return df
