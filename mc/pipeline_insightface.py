"""
Pipeline Pengenalan Wajah Berbasis InsightFace (SCRFD + ArcFace/AdaFace)
Folder: c:\Users\Asus\Documents\FaceRecognition\mc

Fitur:
1. Deteksi wajah kerumunan & wajah kecil (< 50 px) menggunakan SCRFD/RetinaFace.
2. Ekstraksi feature embedding 512-D menggunakan ArcFace (ResNet-50 / IResNet-100).
3. Pencocokan biometrik berbasis Cosine Similarity terhadap multi-era Michael Jackson.
4. Anotasi otomatis & penandaan visual Michael Jackson pada foto kelompok "USA for Africa".

Persyaratan Instalasi:
    pip install insightface onnxruntime opencv-python numpy
    (atau onnxruntime-gpu jika menggunakan NVIDIA GPU)
"""

import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def main():
    mc_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Membaca direktori kerja: {mc_dir}")

    # 1. Inisialisasi InsightFace (Model buffalo_l: SCRFD 10G + ArcFace ResNet50)
    print("\n1. Menginisialisasi model InsightFace (SCRFD Detector + ArcFace Recognizer)...")
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    # det_size=(1024, 1024) memungkinkan deteksi wajah kecil hingga 15-20 piksel di kerumunan
    app.prepare(ctx_id=0, det_size=(1024, 1024))
    print("✓ Model SCRFD & ArcFace siap.")

    # 2. Ekstrak Embedding Referensi Michael Jackson dari foto-foto di folder mc
    reference_files = [
        ("Michael_Jackson_1969.jpg.jpeg", "Era 1969 (Jackson 5)"),
        ("BenMj1.jpg.jpeg", "Era 1972 (Ben)"),
        ("album-cover_offthewall-640x640.jpg.jpeg", "Era 1979 (Off The Wall)"),
        ("gettyimages-1282563440-612x612.jpg.jpeg", "Era 1983 (Thriller)"),
        ("michael-jackson-210aa5866c7d4dd58de8e3af57fe919a.jpg.jpeg", "Era 1984 (Victory / Thriller)"),
        ("images.jpg.jpeg", "Era 1987 (Bad)"),
        ("512227_v9_bb.jpg.jpeg", "Era 2000-an (Invincible)")
    ]

    print("\n2. Mengekstrak 512-D ArcFace embedding dari foto referensi Michael Jackson...")
    mj_templates = []

    for fname, label in reference_files:
        fpath = os.path.join(mc_dir, fname)
        if not os.path.exists(fpath):
            continue
        
        img = cv2.imread(fpath)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) > 0:
            # Ambil wajah terbesar jika ada lebih dari satu
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            emb = normalize_vector(faces[0].embedding)
            mj_templates.append({
                'label': label,
                'filename': fname,
                'embedding': emb
            })
            print(f"  ✓ {label} [{fname}] -> 512-D vector diekstrak")
        else:
            print(f"  ⚠️ Tidak ada wajah terdeteksi pada {fname}")

    if not mj_templates:
        print("❌ Tidak ada template Michael Jackson yang berhasil diekstrak!")
        return

    # Hitung Centroid Vektor (Rata-rata Multi-Era)
    all_embs = np.array([t['embedding'] for t in mj_templates])
    centroid_emb = normalize_vector(np.mean(all_embs, axis=0))
    print(f"✓ Berhasil membangun model ensemble ({len(mj_templates)} era referensi).")

    # 3. Proses Foto Kerumunan "USA for Africa"
    crowd_filename = "we-are-the-world-doc.jpg.jpeg"
    crowd_path = os.path.join(mc_dir, crowd_filename)
    if not os.path.exists(crowd_path):
        print(f"❌ File {crowd_path} tidak ditemukan!")
        return

    print(f"\n3. Memproses foto kelompok: {crowd_filename}...")
    crowd_img = cv2.imread(crowd_path)
    h_orig, w_orig = crowd_img.shape[:2]

    # Deteksi semua wajah di foto kerumunan
    crowd_faces = app.get(crowd_img)
    print(f"✓ Berhasil mendeteksi {len(crowd_faces)} wajah dalam foto kerumunan!")

    # 4. Pencocokan Biometrik terhadap Model Michael Jackson
    best_match = None
    highest_sim = -1.0
    face_matches = []

    # Urutkan wajah dari kiri ke kanan agar penomoran rapi
    crowd_faces = sorted(crowd_faces, key=lambda f: f.bbox[0])

    for idx, face in enumerate(crowd_faces):
        face_emb = normalize_vector(face.embedding)
        
        # Hitung kemiripan terhadap setiap era template
        era_sims = [cosine_similarity(face_emb, t['embedding']) for t in mj_templates]
        max_era_sim = max(era_sims)
        best_era_idx = int(np.argmax(era_sims))
        best_era_label = mj_templates[best_era_idx]['label']

        # Hitung kemiripan terhadap centroid
        centroid_sim = cosine_similarity(face_emb, centroid_emb)
        final_sim = max(max_era_sim, centroid_sim)

        match_info = {
            'index': idx + 1,
            'bbox': [int(v) for v in face.bbox],
            'similarity': final_sim,
            'best_era': best_era_label,
            'is_michael': False
        }

        if final_sim > highest_sim:
            highest_sim = final_sim
            best_match = match_info

        face_matches.append(match_info)

    # Threshold pencocokan ArcFace: umumnya >= 0.45 (cosine sim) sudah menunjukkan orang yang sama
    if best_match and highest_sim >= 0.40:
        best_match['is_michael'] = True
        pct = round(highest_sim * 100, 1)
        print(f"\n🎯 HASIL: Michael Jackson DITEMUKAN pada Kotak #{best_match['index']}!")
        print(f"   - Nilai Cosine Similarity: {highest_sim:.4f} ({pct}%)")
        print(f"   - Paling Mirip dengan: {best_match['best_era']}")
        print(f"   - Bounding Box: {best_match['bbox']}")
    else:
        print("\n⚠️ Michael Jackson tidak mencapai ambang batas kemiripan pada kerumunan ini.")

    # 5. Gambarkan Anotasi Visual pada Gambar
    print("\n4. Menggambar bounding box dan anotasi visual...")
    out_img = crowd_img.copy()

    # Gambar kotak artis lain
    for f in face_matches:
        x1, y1, x2, y2 = f['bbox']
        if not f['is_michael']:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (248, 189, 56), 1)  # Biru muda tipis
            cv2.putText(out_img, f"#{f['index']}", (x1, max(12, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (248, 189, 56), 1, cv2.LINE_AA)

    # Gambar highlight khusus Michael Jackson
    if best_match and best_match['is_michael']:
        x1, y1, x2, y2 = best_match['bbox']
        pct = round(best_match['similarity'] * 100, 1)

        # Kotak hijau tebal
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 230, 118), 3)

        # Sudut kurung (corner brackets)
        c_len = min(15, (x2 - x1) // 4)
        c_thick = 4
        # Top-left
        cv2.line(out_img, (x1, y1), (x1 + c_len, y1), (0, 255, 128), c_thick)
        cv2.line(out_img, (x1, y1), (x1, y1 + c_len), (0, 255, 128), c_thick)
        # Top-right
        cv2.line(out_img, (x2, y1), (x2 - c_len, y1), (0, 255, 128), c_thick)
        cv2.line(out_img, (x2, y1), (x2, y1 + c_len), (0, 255, 128), c_thick)
        # Bottom-left
        cv2.line(out_img, (x1, y2), (x1 + c_len, y2), (0, 255, 128), c_thick)
        cv2.line(out_img, (x1, y2), (x1, y2 - c_len), (0, 255, 128), c_thick)
        # Bottom-right
        cv2.line(out_img, (x2, y2), (x2 - c_len, y2), (0, 255, 128), c_thick)
        cv2.line(out_img, (x2, y2), (x2, y2 - c_len), (0, 255, 128), c_thick)

        # Lencana hijau di atas kepala
        label_text = f"TARGET: MICHAEL JACKSON ({pct}%)"
        (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_y1 = max(40, y1 - 25)
        badge_x1 = max(5, x1 - 10)
        cv2.rectangle(out_img, (badge_x1, badge_y1 - t_h - 6), (badge_x1 + t_w + 12, badge_y1 + 4), (0, 200, 83), -1)
        cv2.putText(out_img, label_text, (badge_x1 + 6, badge_y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Panah penunjuk ke arah kepala
        cv2.arrowedLine(out_img, (x1 + (x2 - x1)//2, badge_y1 + 10), (x1 + (x2 - x1)//2, y1 - 2), (0, 255, 128), 2, tipLength=0.3)

    # Tambahkan Header Bar & Footer Summary
    overlay = out_img.copy()
    cv2.rectangle(overlay, (0, 0), (w_orig, 42), (15, 23, 42), -1)
    cv2.rectangle(overlay, (0, h_orig - 42), (w_orig, h_orig), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, out_img, 0.15, 0, out_img)

    cv2.putText(out_img, "INSIGHTFACE PIPELINE: SCRFD (1024x1024) + ARCFACE (RESNET-50) 512-D", (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)
    
    if best_match and best_match['is_michael']:
        pct = round(best_match['similarity'] * 100, 1)
        footer_text = f"TERIDENTIFIKASI: Michael Jackson pada Kotak #{best_match['index']} (Kemiripan: {pct}% - {best_match['best_era']})"
        cv2.putText(out_img, footer_text, (15, h_orig - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)

    # 6. Simpan Hasil
    out_filename = "hasil_insightface_we_are_the_world.jpg"
    out_path = os.path.join(mc_dir, out_filename)
    cv2.imwrite(out_path, out_img)
    print(f"\n✅ Selesai! Hasil gambar anotasi disimpan ke: {out_path}")

if __name__ == "__main__":
    main()
