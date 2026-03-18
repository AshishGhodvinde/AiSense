import cv2
import numpy as np
import io
from PIL import Image, ImageChops, ImageEnhance, ExifTags

def compute_ela(image_bytes, quality=90):
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        temp_io = io.BytesIO()
        original.save(temp_io, 'JPEG', quality=quality)
        temp_io.seek(0)
        
        resaved = Image.open(temp_io)
        ela_image = ImageChops.difference(original, resaved)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1
            
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        return np.array(ela_image)
    except Exception as e:
        print(f"ELA Error: {e}")
        return None

def compute_high_freq_noise(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance
    except Exception as e:
        print(f"Noise Error: {e}")
        return 0.0

def scan_c2pa(image_bytes):
    # Quick bytes search for 'JUMBF' or 'c2pa'
    if b'JUMBF' in image_bytes or b'c2pa' in image_bytes:
        return True
    return False

def get_exif_data(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img._getexif()
        metadata = {}
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                metadata[decoded] = value
        
        software = None
        if 'Software' in metadata:
            software = str(metadata['Software'])
        elif 'software' in img.info:
            software = str(img.info['software'])
            
        return metadata, software, img.size
    except Exception as e:
        return {}, None, (0,0)

def compute_exif_heuristics(image_bytes, filename=""):
    metadata, software, dims = get_exif_data(image_bytes)
    has_c2pa = scan_c2pa(image_bytes)
    
    reasons = []
    
    if has_c2pa:
        reasons.append("C2PA provenance data detected (JUMBF).")
    
    has_make = 'Make' in metadata
    has_model = 'Model' in metadata
    has_lens = 'LensModel' in metadata
    exposure_keys = ['FNumber', 'ExposureTime', 'ISOSpeedRatings', 'ISO', 'FocalLength']
    has_exposure = any(k in metadata for k in exposure_keys)
    has_date = 'DateTimeOriginal' in metadata
    has_gps = 'GPSInfo' in metadata
    
    device_evidence = 0
    if has_make and has_model: device_evidence += 2
    if has_exposure: device_evidence += 1
    if has_gps: device_evidence += 1
    if has_lens or has_date: device_evidence += 0.5
    
    ai_hints = ['midjourney','stability','stable diffusion','sdxl','comfyui','invokeai','automatic1111',
      'dalle','openai','firefly','bing image creator','leonardo ai','playground ai','ideogram',
      'pixray','nightcafe','craiyon','gen-2','sd next','flux','recraft']
      
    software_str = (software or "").lower()
    has_ai_tool = any(hint in software_str for hint in ai_hints)
    
    if has_ai_tool:
        reasons.append(f'Software indicates AI generator: "{software}"')
        return {'isAi': True, 'confidence': 90, 'reasons': reasons}
        
    exif_present = len(metadata) > 0
    
    if exif_present and device_evidence > 0:
        present = []
        if has_make: present.append(f"Make: {metadata.get('Make')}")
        if has_model: present.append(f"Model: {metadata.get('Model')}")
        if has_exposure: present.append('Exposure data' if has_exposure else '')
        if has_lens: present.append('Lens info')
        if has_gps: present.append('GPS')
        if has_date: present.append('Date')
        
        reasons.append(f"Camera metadata found ({', '.join(present)}).")
        conf = int(min(96.0, 70.0 + float(device_evidence) * 8.0))
        return {'isAi': False, 'confidence': conf, 'reasons': reasons}
        
    messaging_indicators = ['whatsapp', 'wa', 'telegram', 'signal', 'messenger', 'wechat', 'snapchat', 'instagram']
    fname_lower = filename.lower()
    looks_messaging = any(w in fname_lower for w in messaging_indicators) or fname_lower.startswith('img-') or fname_lower.startswith('img_') or fname_lower.startswith('pxl_')
    
    w, h = dims
    min_side = min(w, h)
    max_side = max(w, h)
    aspect = max_side / min_side if min_side > 0 else 0
    common_messaging_max = max_side > 600 and max_side <= 2048
    common_aspect = aspect > 1.2 and aspect < 2.0
    
    if not exif_present and (looks_messaging or (common_messaging_max and common_aspect)):
        reasons.append('No EXIF, but dimensions/name suggest messaging app re-encode (likely real photo).')
        return {'isAi': False, 'confidence': 80, 'reasons': reasons}
        
    if "png" in filename.lower() and not exif_present:
        reasons.append('PNG has no EXIF; common for AI exports.')
        
    reasons.append('Insufficient camera metadata (make/model/exposure/date/GPS).')
    
    edit_hints = ['photoshop', 'lightroom', 'gimp']
    if any(h in software_str for h in edit_hints):
        reasons.append('Edited in an image editor (not conclusive).')
        
    conf_base = int(min(88.0, 60.0 + (2.0 - min(2.0, float(device_evidence))) * 10.0))
    return {'isAi': True, 'confidence': conf_base, 'reasons': reasons}
