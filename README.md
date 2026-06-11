# Real-time face recognition system

Identifies people from a live webcam feed, predicts their age and gender, and persists identity data across sessions. Built with Python, OpenCV, and the `face_recognition` library.

## What it does

The system opens your webcam and runs three things simultaneously: identity matching against a known-faces database, age estimation across eight age brackets, and binary gender classification. Recognition results are overlaid on the frame in real time, along with a confidence score for each match.

Face encodings are stored as mathematical vectors, not raw images. That means the database file stays small and the comparisons are fast. Encodings persist in `face_encodings_database.pkl` between sessions, so you don't re-register faces every time you run it.

To keep the frame rate usable, the system runs face recognition every 5 frames and caches the results in between. Age and gender inference uses Caffe models loaded through OpenCV's DNN module and falls back gracefully if the model files aren't found.

## Files

| File | Purpose |
|------|---------|
| `Face recognition 4.py` | Main script |
| `age_deploy.prototxt` | Age model architecture |
| `age_net.caffemodel` | Age model weights (~45MB, requires Git LFS) |
| `gender_deploy.prototxt` | Gender model architecture |
| `gender_net.caffemodel` | Gender model weights (~45MB, requires Git LFS) |
| `face_encodings_database.pkl` | Persisted face encodings (auto-generated on first run) |

> **Note on model files:** Both `.caffemodel` files are ~45MB each. GitHub's standard file size limit is 100MB, but you should use [Git LFS](https://git-lfs.github.com/) for them anyway to keep repo clones fast. Alternatively, download them from [this Caffe model repository](https://github.com/spmallick/learnopencv/tree/master/AgeGender) and place them in the project root.

## Setup

**Prerequisites:** Python 3.8+, a working webcam, and `cmake` installed before `dlib` (required by `face_recognition`).

```bash
pip install opencv-python face_recognition numpy
```

On some systems, installing `dlib` needs:

```bash
pip install cmake dlib
pip install face_recognition
```

Then run:

```bash
python "Face recognition 4.py"
```

## Controls

Once the window opens:

| Key | Action |
|-----|--------|
| `A` | Register the visible face under a name you type |
| `L` | List all registered names in the database |
| `R` | Remove a person from the database by name |
| `C` | Clear the entire database and delete the pickle file |
| `T` | Adjust the recognition tolerance (0.1 = strict, 1.0 = lenient) |
| `S` | Save a screenshot to the current directory |
| `Q` | Quit |

The default tolerance is `0.6`. If you're getting false matches, lower it to `0.5` or `0.45`.

## How the recognition works

When you press `A`, the script calls `face_recognition.face_encodings()` on the detected face and stores the resulting 128-dimensional vector alongside the name you enter. On each processing frame, it computes the Euclidean distance between the incoming face encoding and all stored encodings and picks the closest match below the tolerance threshold. The confidence score shown on screen is `(1 - distance) * 100`.

The system checks for duplicate registrations before adding a new entry. If the incoming encoding matches an existing one, it asks whether you want to update the name instead.

## Project context

Built as the computer vision component of a 3-month AI internship at SmartED Innovations, assessed by Skill India and NSDC (Jul-Oct 2025).

## Dependencies

- `opencv-python`
- `face_recognition`
- `numpy`
- `dlib` (installed automatically with `face_recognition`)
