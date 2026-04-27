import pandas as pd

class DataLoader:
    """
    Lớp tải và quản lý dữ liệu cho hệ thống.
    """
    def __init__(self):
        pass

    def generate_dummy_data(self):
        """
        Tạo dữ liệu giả lập (Dummy Data) dạng DataFrame gồm thông tin các món ăn.
        Các đặc trưng bao gồm:
        - is_spicy: 1 (Cay), 0 (Không cay)
        - is_vegan: 1 (Chay), 0 (Mặn)
        - is_sweet: 1 (Ngọt), 0 (Không ngọt)
        - price_range: 1 (Rẻ), 2 (Vừa), 3 (Đắt)
        - region_code: 1 (Miền Bắc), 2 (Miền Trung), 3 (Miền Nam), 0 (Quốc tế)
        """
        data = {
            'food_id': range(1, 11),
            'food_name': ['Phở bò', 'Bún chả', 'Salad chay', 'Canh chua', 'Chè trôi nước', 
                          'Gà rán', 'Lẩu thái cay', 'Đậu hũ tứ xuyên', 'Bánh xèo', 'Gỏi cuốn chay'],
            'is_spicy': [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            'is_vegan': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            'is_sweet': [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            'price_range': [2, 2, 1, 2, 1, 3, 3, 2, 2, 1], 
            'region_code': [1, 1, 0, 3, 1, 0, 0, 0, 3, 3]  
        }
        df = pd.DataFrame(data)
        return df
