# proxi-track

# Hand Tracking Proximity Warning System (POC)
A real-time computer vision prototype that detects a user’s hand/fingertips using classical image-processing techniques — **without MediaPipe, OpenPose, or pose-detection APIs** — and triggers visual warnings when the hand approaches a virtual object on the screen.

This POC is built as part of the **Arvyax Internship Assignment**, demonstrating:
- Real-time hand/fingertip tracking  
- Virtual on-screen boundary  
- Distance-based SAFE / WARNING / DANGER states  
- Clear visual feedback overlay  
- CPU-only execution ≥ 8 FPS  

---

## 🎥 Demo Video
Watch the demo here:  

[![Demo Video](demo_thumbnail.png)](https://github.com/rohankharche34/proxi-track/releases/tag/v1.0)


## 🚀 Features

### ✅ **1. Real-Time Fingertip Tracking (No ML APIs)**
Implemented using classical CV methods:
- HSV-based skin color segmentation  
- Contours  
- Convex Hull  
- Convexity Defects  
- Fingertip clustering  

This enables detection of **multiple fingertips**, ensuring even a single finger approaching the boundary triggers detection.

---

### ✅ **2. Virtual Object / Boundary**
A rectangular region on the right-hand side of the screen acts as a **danger zone**.  
Its color changes based on distance classification.

---

### ✅ **3. Distance-Based State Logic**
For every frame:
1. Detect fingertips  
2. Compute distance from each fingertip to the virtual box  
3. Choose the minimum distance  
4. Classify into:

| State     | Description |
|----------|-------------|
| **SAFE** | Hand is comfortably far |
| **WARNING** | Hand approaching box |
| **DANGER** | Fingertip touching/very close to box |

Threshold values are configurable.

---

### ✅ **4. Visual Overlays**
- Tracking dots on detected fingertip locations  
- Bounding box around the virtual object  
- Top-left state indicator  
- Center-screen flashing **DANGER DANGER** message  
- FPS counter  

---

### ✅ **5. Real-Time Performance**
The prototype achieves:
- **8–20 FPS on CPU-only**
- No GPU or heavy ML models used
- Lightweight OpenCV + NumPy pipeline  

---

## 🛠️ Tech Stack

- Python 3.x  
- OpenCV  
- NumPy  
- Classical CV algorithms (no deep learning APIs)  

---

## 📦 Installation

### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/hand-tracking-poc
cd hand-tracking-poc
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### ▶️ Running the Prototype
```bash
python main.py
```
