import cv2
import numpy as np

def extract_frequency_features(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, [0.0, 0.0, 0.0]

        # Apply FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2

        # 1. Mean frequency energy
        mean_energy = np.mean(magnitude_spectrum)
        
        # 2. High-frequency ratio
        # Mask out the low frequencies
        r = min(h, w) // 4
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x*x + y*y <= r*r
        
        low_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_energy = total_energy - low_freq_energy
        high_freq_ratio = high_freq_energy / (total_energy + 1e-5)
        
        # 3. Radial distribution
        radii_means = []
        for radius in range(10, min(h, w) // 2, 10):
            r_mask = (x*x + y*y >= radius*radius) & (x*x + y*y < (radius+10)**2)
            if np.sum(r_mask) > 0:
                radii_means.append(np.mean(magnitude_spectrum[r_mask]))
                
        radial_variance = np.var(radii_means) if radii_means else 0.0
        
        # Calculate AI likelihood score with a neutral baseline of 0.5
        ai_probability = 0.5
        
        if high_freq_ratio > 0.6:
            ai_probability += 0.15
        elif high_freq_ratio < 0.15:
            ai_probability -= 0.15
            
        if radial_variance > 500:
            ai_probability += 0.20
        elif radial_variance < 100:
            ai_probability -= 0.10
            
        if mean_energy < 50:
            ai_probability += 0.10
        elif mean_energy > 200:
            ai_probability -= 0.10
            
        feature_vector = [float(mean_energy), float(high_freq_ratio), float(radial_variance)]
        ai_probability = min(0.95, max(0.05, ai_probability))
        
        return ai_probability, feature_vector
        
    except Exception as e:
        print(f"Frequency Analysis Error: {e}")
        return 0.0, [0.0, 0.0, 0.0]
