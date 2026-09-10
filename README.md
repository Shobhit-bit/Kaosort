# Face Sorting System

A Streamlit app that scans a folder of photos, detects and filters faces, then clusters them into folders by identity using face embeddings — handy for sorting event or group photos by the people in them.

## How it works

1. **Scan** — Upload a zip of photos. The app unzips them to a temporary folder, runs face detection with InsightFace (`buffalo_l` model), and keeps only faces that pass the quality filters (blur, confidence, size ratio). Photos containing a valid face are copied to `<output_path>/embeds`.
2. **Cluster** — Once scanning finishes, choose how many clusters (folders) to create. Agglomerative clustering (cosine distance, average linkage) groups the extracted face embeddings, and each photo is copied into `<output_path>/Hito_wa_<cluster_id>/`.

## Features

- GPU-accelerated face detection/embedding via InsightFace (CUDA, with automatic CPU fallback)
- Adjustable quality filters:
  - **Blur threshold** — Laplacian variance of the face crop; higher = stricter
  - **Face detection confidence** — minimum detector score; higher = stricter
  - **Min face-to-photo ratio** — rejects faces that only occupy a small part of the frame
- Upload photos as a single zip — no manual unzipping needed
- Unsupervised clustering into a user-specified number of folders

## Requirements

- Python 3.9+
- NVIDIA GPU + CUDA drivers (optional — falls back to CPU automatically)

### Python packages

```
streamlit
insightface
opencv-python
numpy
scikit-learn
onnxruntime-gpu   # or plain onnxruntime for CPU-only setups
```

Install with:

```bash
pip install streamlit insightface opencv-python numpy scikit-learn onnxruntime-gpu
```

## Usage

Run the app with:

```bash
python3 -m streamlit run app.py --server.port 6969
```

Then, in the browser UI:

1. Upload a `.zip` file containing the photos you want to sort.
2. Enter an **output directory** path — it must already exist and be writable by the machine running the app.
3. Adjust the quality filters if needed:
   - **Blur thres** (default 85) — raise to reject blurrier faces
   - **Face_detection_confidence** (default 0.85) — raise to reject low-confidence detections
   - **Min_Face_to_photo_ratio** (default 0.06) — raise to reject faces that are too small relative to the photo
4. Click **STEP 1: Scan Zip** to extract embeddings for all qualifying faces.
5. Once scanning completes, set **Number of folders to create** to your best estimate of how many distinct people are in the photo set.
6. Click **CLUSTER NOOOOWWWWW!!!!** to run clustering and sort photos.
7. Sorted photos land in `<output_path>/Hito_wa_<cluster_id>/`, one folder per detected identity.

## Project structure

```
.
├── app.py    # Streamlit UI and end-to-end workflow
└── core.py   # Face quality filtering, embedding extraction, and clustering logic
```

## Known limitations

- **No path validation** — the output directory field is free text; make sure it exists and is writable before scanning.
- **Cluster count is manual** — `AgglomerativeClustering` needs a fixed number of clusters. If your guess doesn't match the actual number of people, faces may get merged together or one person may get split across folders.
- **`embed_dir` cleanup on cluster step** — `os.rmdir(embed_dir)` at the end of the clustering step assumes `embed_dir` is defined and empty. Because Streamlit reruns the whole script on every interaction, this variable may not be set in the run where the cluster button is pressed, which can raise a `NameError`. Recomputing `embed_dir = os.path.join(output_path, "embeds")` right before this call (and using `shutil.rmtree` in case files remain) would make this more robust.
- **No deduplication** — re-running a scan on overlapping photo sets doesn't check for duplicates.
- **One face per person assumption** — photos with multiple faces will contribute one embedding per detected face, so a single photo can end up copied into more than one cluster folder.
