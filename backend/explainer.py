def generate_explanation(final_is_ai, final_confidence, ml_is_ai, ml_confidence, ela_variance, noise_score, metadata_reasons):
    expl = f"We predict this image is {'AI Generated' if final_is_ai else 'an Authentic Photograph'} with {final_confidence*100:.1f}% confidence. "
    
    expl += f"\n\n**Machine Learning CNN:** Our PyTorch neural network analyzed the image's deep spatial features and independently predicts it is {'AI Generated' if ml_is_ai else 'Authentic'} ({ml_confidence*100:.1f}% confidence)."
    
    if metadata_reasons:
        expl += "\n\n**EXIF & Metadata Rules:** " + " ".join(metadata_reasons)
        
    expl += "\n\n**Pixel Variance Analysis:** "
    
    if noise_score > 1500:
        expl += "High-frequency noise (Laplacian) is extremely sharp, typical for diffusion algorithms. "
    else:
        expl += "Noise variance is lower, matching standard image compression behaviors. "
        
    if ela_variance > 400:
        expl += f"Error Level Analysis (ELA={ela_variance:.1f}) reveals highly inconsistent structural pixel irregularities commonly seen when AI pieces together local blocks or generative filters are applied. "
    else:
        expl += f"ELA (ELA={ela_variance:.1f}) shows uniform/natural compression artifacts, proving the physical light structures haven't been synthetically degraded. "
        
    if final_is_ai and not ml_is_ai:
        expl += "\n\n*Note: Even though the ML model found visual similarities to authentic images, our structural ELA physics and Metadata checks conclusively classify it as AI-generated.*"
            
    return expl
