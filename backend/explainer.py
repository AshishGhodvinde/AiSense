def generate_explanation(final_is_ai, final_confidence, ml_is_ai, ml_confidence, freq_prob, noise_prob, artifact_prob, metadata_reasons, ela_variance=0.0):
    verdict = 'AI Generated' if final_is_ai else 'an Authentic Photograph'
    expl = f"We predict this image is **{verdict}** with {final_confidence*100:.1f}% confidence.\n\n"
    
    # ML signal
    ml_label = 'AI Generated' if ml_is_ai else 'Authentic'
    expl += f"**AI Model Ensemble:** Our multi-model detector classifies this image as {ml_label} ({ml_confidence*100:.1f}% confidence). "
    expl += "Two independently trained HuggingFace ViT models (trained on real Midjourney/Stable Diffusion/DALL-E imagery) contributed to this score.\n\n"
    
    # ELA
    if ela_variance > 300:
        expl += f"**Error Level Analysis (ELA={ela_variance:.0f}):** Strong regional inconsistencies detected. Different image regions show drastically different compression artifacts — a hallmark of AI inpainting or partial regeneration where only certain areas were synthetically altered.\n\n"
    elif ela_variance > 100:
        expl += f"**Error Level Analysis (ELA={ela_variance:.0f}):** Moderate structural irregularities present, suggesting possible AI-assisted editing or enhancement.\n\n"
    else:
        expl += f"**Error Level Analysis (ELA={ela_variance:.0f}):** Compression artifacts are uniform across the image, consistent with an unaltered photograph.\n\n"
    
    # EXIF
    if metadata_reasons:
        expl += "**EXIF & Metadata:** " + " ".join(metadata_reasons) + "\n\n"
    
    # Signal analysis
    signals = []
    if freq_prob > 0.6:
        signals.append("high-frequency spectral anomalies typical of generative diffusion models")
    if noise_prob > 0.6:
        signals.append("unnaturally smooth noise patterns inconsistent with physical camera sensors")
    if artifact_prob > 0.6:
        signals.append("over-smooth local regions and repeated texture patterns common in AI synthesis")
    
    if signals:
        expl += "**Signal Findings:** This image exhibits " + ", and ".join(signals) + "."
    
    return expl
