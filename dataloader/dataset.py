import numpy as np

def processing_data(features):
    """
    Hàm xử lý và chuẩn hóa dữ liệu pose.
    Đầu vào là tọa độ xy của các keypoints.
    """

    def scale_pose(xy):
        """
        Chuẩn hóa các điểm pose bằng cách co giãn theo giá trị min/max của mỗi pose.
        Mục tiêu là đưa toàn bộ bộ xương vào một hộp [-1, 1] mà vẫn giữ được tỷ lệ.
        
        Input: xy - có dạng (frames, parts, 2)
        """
        # Xử lý trường hợp đầu vào chỉ có 1 frame
        if xy.ndim == 2:
            xy = np.expand_dims(xy, 0)

        # Tìm tọa độ min và max trên TẤT CẢ các keypoints (axis=1) cho mỗi frame
        # ĐÂY LÀ THAY ĐỔI QUAN TRỌNG: axis=1 thay vì axis=2
        xy_min = np.nanmin(xy, axis=1, keepdims=True)
        xy_max = np.nanmax(xy, axis=1, keepdims=True)

        # Tính toán khoảng giá trị, xử lý trường hợp chia cho 0
        xy_range = xy_max - xy_min
        xy_range = np.where(xy_range == 0, 1.0, xy_range)

        # Chuẩn hóa về khoảng [-1, 1]
        xy_normalized = ((xy - xy_min) / xy_range) * 2 - 1

        # Xử lý các giá trị NaN/inf còn sót lại sau phép tính
        xy_normalized = np.nan_to_num(xy_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Đảm bảo các giá trị nằm trong khoảng hợp lệ
        xy_normalized = np.clip(xy_normalized, -1.0, 1.0)

        return xy_normalized.squeeze()

    # Áp dụng hàm chuẩn hóa
    features_scaled = scale_pose(features)
    
    return features_scaled


def processing_data_legacy(features):
    """
    Hàm xử lý cũ - giữ lại để tương thích nếu cần.
    """
    def scale_pose(xy):
        if xy.ndim == 2:
            xy = np.expand_dims(xy, 0)
        xy_min = np.nanmin(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
        xy_max = np.nanmax(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
        xy = (xy - xy_min) / (xy_max - xy_min) * 2 - 1
        return xy

    features = scale_pose(features)
    return features