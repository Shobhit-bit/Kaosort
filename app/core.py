import os
import cv2
import shutil
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def quality_filter(img, face, blur_thresh, conf_thresh, ratio_thresh):
    if face.det_score < conf_thresh:
        return False
    
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    y1, y2 = max(0, y1), min(img.shape[0], y2)
    x1, x2 = max(0, x1), min(img.shape[1], x2)
    face_crop = img[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return False
    
    # Area ratio check
    face_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]
    if (face_area / img_area) < ratio_thresh:
        return False
    
    # Blur check
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thresh:
        return False
        
    return True

def extract_embeddings(model, input_dir, blur, conf, ratio, progress_callback=None):
    all_embeds = []
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        faces = model.get(img)
        for face in faces:
            if quality_filter(img, face, blur, conf, ratio):
                all_embeds.append({
                    "path": img_path,
                    "embed": face.embedding,
                    "name": img_name
                })
        
        if progress_callback:
            progress_callback(i + 1, len(image_files))
            
    return all_embeds

def run_clustering(all_embeds, n_clusters):
    if not all_embeds:
        return []
    
    embed_mat = np.array([f["embed"] for f in all_embeds])
    cluster_algo = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric="cosine", 
        linkage="average"
    )
    labels = cluster_algo.fit_predict(embed_mat)
    
    for i, face_data in enumerate(all_embeds):
        face_data["id"] = labels[i]
    return all_embeds
