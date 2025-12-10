from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <--- Tambah ini
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# --- BAGIAN INI SANGAT PENTING ---
# Mengizinkan siapa saja (termasuk laptop teman) untuk akses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" artinya semua boleh masuk
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------

# 2. Load Model yang sudah disimpan tadi
# Pastikan file .pkl ada di folder yang sama
model = joblib.load('model_sampah.pkl')

# 3. Definisikan Format Input (Data yang dikirim Frontend)
# Contoh: Model prediksi harga rumah berdasarkan Luas dan Jumlah Kamar
class InputData(BaseModel):
    luas_tanah: float
    jumlah_kamar: int

# 4. Buat Endpoint (Pintunya)
@app.post("/predict")
def predict(data: InputData):
    # Ubah data input menjadi format yang dimengerti model (array 2D)
    features = np.array([[data.luas_tanah, data.jumlah_kamar]])
    
    # Lakukan prediksi menggunakan model
    prediksi = model.predict(features)
    
    # Kembalikan hasil ke Frontend dalam bentuk JSON
    return {
        "status": "sukses",
        "prediksi_harga": prediksi[0]
    }

# Endpoint Root (Cek apakah server jalan)
@app.get("/")
def home():
    return {"message": "API Model Siap Digunakan!"}