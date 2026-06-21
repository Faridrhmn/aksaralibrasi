# Aksaralibrasi - Pengenalan Aksara Jawa

Aksaralibrasi adalah aplikasi web berbasis Python yang dirancang untuk mendeteksi dan mengenali karakter Aksara Jawa dari sebuah citra digital. Aplikasi ini memanfaatkan **Freeman Chain Code (FCC)** untuk ekstraksi fitur atau ciri dari karakter dan **Support Vector Machine (SVM)** sebagai model klasifikasinya.

## Fitur Utama

- **Upload & Deteksi**: Pengguna dapat mengunggah gambar berisi tulisan Aksara Jawa untuk dianalisis.
- **Sampel Prediksi**: Menyediakan gambar sampel bawaan untuk mempermudah uji coba secara langsung tanpa harus mengunggah gambar.
- **Image Preprocessing**: Menerapkan berbagai tahapan pengolahan citra seperti *grayscaling*, *thresholding*, *morphological operations*, hingga *thinning* (Zhang-Suen) sebelum proses pengenalan.
- **Visualisasi Hasil**: Menampilkan tidak hanya hasil akhir, namun juga visualisasi setiap karakter yang tersegmentasi (ROI) beserta tahapan preprocessingnya.

## Teknologi yang Digunakan

- **Backend / Web Framework**: Python (Flask)
- **Computer Vision & Image Processing**: OpenCV, NumPy, Matplotlib
- **Machine Learning**: Scikit-Learn (SVM), Joblib

## Instalasi dan Cara Penggunaan

1. **Clone repository ini** (atau unduh *source code*)
   ```bash
   git clone <url-repo-anda>
   cd bismillah_skripsi
   ```

2. **Buat dan aktifkan virtual environment** (direkomendasikan)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   # venv\Scripts\activate   # Untuk Windows
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**
   ```bash
   python app.py
   ```
   Aplikasi akan berjalan di `http://127.0.0.1:5000/` atau `http://localhost:5000/`.

5. **Akses lewat Browser**
   Buka alamat URL di atas pada browser Anda. Anda bisa mengunggah gambar atau mengklik salah satu gambar sampel untuk memulai prediksi Aksara Jawa.

## Struktur Direktori

- `app.py`: Main script untuk menjalankan server Flask dan logika OCR/pengolahan citra.
- `requirements.txt`: Daftar dependensi Python yang dibutuhkan.
- `joblibs/`: Menyimpan model machine learning (SVM) dan *scaler* yang sudah dilatih (pretrained).
- `templates/`: Folder berisi file HTML (`index.html`) untuk tampilan antarmuka (frontend).
- `static/`: Folder untuk file statis seperti CSS dan JavaScript.
- `history/`: Tempat penyimpanan sementara gambar hasil upload, proses *bounding box*, dan *output*.
- `sample/`: Folder berisi gambar-gambar sampel untuk diuji coba.

## Demo Projek
https://bachelor-thesis.faridrhmn.my.id/