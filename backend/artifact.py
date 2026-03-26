import cv2
import numpy as np

def detect_artifacts(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, {}

        h, w = img.shape
        
        # 1. Over-smooth regions (low variance patches)
        block_size = 32
        smooth_blocks = 0
        total_blocks = 0
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img[y:y+block_size, x:x+block_size]
                if np.var(block) < 5.0:
                    smooth_blocks += 1
                total_blocks += 1
                
        smooth_ratio = smooth_blocks / max(1, total_blocks)
        
        # 2. Edge inconsistency
        edges = cv2.Canny(img, 100, 200)
        edge_density = np.sum(edges > 0) / (h * w)
        contrast = np.std(img)
        edge_density_irregularity = abs((contrast / 64.0) - (edge_density * 10))
        
        # 3. Repeated textures
        img_float = img.astype(np.float32)
        shift_x = np.roll(img_float, 8, axis=1)
        shift_y = np.roll(img_float, 8, axis=0)
        diff_x = np.abs(img_float - shift_x).mean()
        diff_y = np.abs(img_float - shift_y).mean()
        
        texture_repetition_score = 255.0 / (diff_x + diff_y + 1e-5)
        
        # Calculate AI probability score with neutral baseline
        ai_probability = 0.5
        
        if smooth_ratio > 0.4:
            ai_probability += 0.25
        elif smooth_ratio < 0.1:
            ai_probability -= 0.15
            
        if edge_density_irregularity > 2.0:
            ai_probability += 0.20
            
        if texture_repetition_score > 30: 
            ai_probability += 0.20
            
        ai_probability = min(0.95, max(0.05, ai_probability))
        return ai_probability, {
            "smooth_ratio": float(smooth_ratio),
            "edge_irregularity": float(edge_density_irregularity),
            "texture_repetition": float(texture_repetition_score)
        }
        
    except Exception as e:
        print(f"Artifact Detection Error: {e}")
        return 0.0, {}
