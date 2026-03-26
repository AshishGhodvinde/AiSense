import cv2
import numpy as np

def analyze_noise_pattern(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, {}

        # 1. Extract noise residual
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        img_16 = img.astype(np.int16)
        blurred_16 = blurred.astype(np.int16)
        noise_residual = np.abs(img_16 - blurred_16)
        
        # 2. Compute noise variance
        noise_variance = np.var(noise_residual)
        
        # 3. Compute noise entropy
        hist, _ = np.histogram(noise_residual, bins=256, range=(0, 256))
        # custom entropy to avoid scipy dependency
        prob = hist / (hist.sum() + 1e-7)
        prob = prob[prob > 0]
        noise_entropy = -np.sum(prob * np.log2(prob))
        
        # 4. Spatial consistency
        h, w = noise_residual.shape
        block_size = min(h, w) // 8
        if block_size < 1:
            block_size = 1
            
        block_variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_residual[y:y+block_size, x:x+block_size]
                block_variances.append(np.var(block))
                
        spatial_consistency = np.var(block_variances) if block_variances else 0.0
        
        # Evaluate AI probability with neutral baseline
        ai_probability = 0.5
        
        # very low noise variance = too smooth = AI trait
        if noise_variance < 5.0:
            ai_probability += 0.30
        elif noise_variance < 10.0:
            ai_probability += 0.15
        elif noise_variance > 50.0:
            ai_probability -= 0.20 # high noise is typically physical sensor trait
            
        if noise_entropy < 2.5:
            ai_probability += 0.20
        elif noise_entropy > 4.0:
            ai_probability -= 0.20
            
        if spatial_consistency < 50.0:
            ai_probability += 0.10
            
        ai_probability = min(0.95, max(0.05, ai_probability))
        
        return ai_probability, {
            "noise_variance": float(noise_variance),
            "noise_entropy": float(noise_entropy), 
            "spatial_consistency": float(spatial_consistency)
        }
    except Exception as e:
        print(f"Noise Analysis Error: {e}")
        return 0.0, {}
