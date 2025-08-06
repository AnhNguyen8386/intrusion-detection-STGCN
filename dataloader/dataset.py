import numpy as np  
  
  
def processing_data(features):  
  
    # ***************************************** NORMALIZE ************************************  
    def scale_pose(xy):  
        """  
        Normalize pose points by scale with max/min value of each pose.  
        xy : (frames, parts, xy) or (parts, xy)  
        """  
        if xy.ndim == 2:  
            xy = np.expand_dims(xy, 0)  
          
        # Calculate min/max across parts for each frame  
        xy_min = np.nanmin(xy, axis=1, keepdims=True) # Keep dimension for broadcasting  
        xy_max = np.nanmax(xy, axis=1, keepdims=True) # Keep dimension for broadcasting  
          
        # Avoid division by zero for constant values  
        denominator = (xy_max - xy_min)  
        denominator[denominator == 0] = 1e-8 # Small epsilon to prevent division by zero  
          
        xy = (xy - xy_min) / denominator * 2 - 1  
        return xy.squeeze() # Remove single-dimensional entries if any  
  
    features = scale_pose(features)  
    return 