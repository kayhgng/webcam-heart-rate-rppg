# webcam-heart-rate-rppg
# Webcam Heart Rate Monitor (rPPG)

A **non-contact heart rate estimation system** based on **remote Photoplethysmography (rPPG)** using a standard webcam.
This project extracts subtle color variations from facial skin regions to estimate heart rate **without any physical sensors**.

> ⚠ **Important Disclaimer**
> This project provides an *estimated* heart rate for research, educational, and demonstration purposes only.
> It is **NOT a medical device** and **MUST NOT** be used for clinical diagnosis, treatment, or medical decision-making.

---

## 🔬 Scientific Background

When the heart pumps blood, the blood volume beneath the skin changes slightly. These changes cause **minute variations in skin color**, especially in the **green color channel**, which can be captured by a camera.

This technique is known as **remote Photoplethysmography (rPPG)** and has been validated in academic research since 2008.

Key references:

* Verkruysse et al., *Remote plethysmographic imaging using ambient light*, 2008
* Poh et al., MIT Media Lab, 2010–2011

---

## ⚙️ How It Works

1. **Face Detection** using Haar Cascades (OpenCV)
2. **Region of Interest (ROI)** selection (upper facial area / forehead)
3. **Green-channel signal extraction** from the ROI
4. **Temporal band-pass filtering** (0.7–4.0 Hz)
5. **Peak detection** to estimate heart rate (BPM)

---

## ✨ Features

* Real-time webcam processing
* Non-contact heart rate estimation
* Signal visualization (raw vs filtered)
* Robust band-pass filtering
* No external sensors required

---

## 🧪 Limitations

This system is subject to inherent limitations of rPPG technology:

* Sensitive to head movement
* Sensitive to lighting conditions
* Reduced accuracy for darker skin tones or low-quality cameras
* Fixed FPS assumption (approx. 30 FPS)

Typical accuracy (under good conditions): **±3–6 BPM**
Accuracy decreases with motion or poor lighting.

---

## 🛠 Requirements

* Python 3.8+
* OpenCV
* NumPy
* SciPy
* Matplotlib

Install dependencies:

```bash
pip install opencv-python numpy scipy matplotlib
```

---

## ▶️ Usage

Run the script:

```bash
python heart_rate_monitor.py
```

Tips for best results:

* Sit still facing the camera
* Use stable, diffuse lighting
* Avoid strong shadows or flickering lights

Press any key or click the window to exit.

---

## 📁 Project Structure

```text
heart_rate_monitor.py   # Main application
README.md               # Project documentation
```

---

## 🚀 Future Improvements

* Motion compensation (optical flow)
* Adaptive ROI selection
* Advanced rPPG methods (CHROM / POS)
* Skin tone normalization
* Machine learning-based refinement
* Validation against pulse oximeter / ECG

---

## 📜 License

This project is released under the **Appache License** (see `LICENSE` file).

---

## 👤 Author

Developed by **Alikay_h**
2025

---

## ⭐ Acknowledgments

Inspired by academic research in computer vision and biomedical signal processing.

If you find this project useful, consider starring the repository ⭐
