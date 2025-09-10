from flask import Flask, render_template, request
import pickle
import numpy as np
import math

app = Flask(__name__)

# Load the model and scaler
with open("random_forest_model.pkl", "rb") as f_model, \
     open("scaler_x_minmax.pkl", "rb") as f_scaler_x, \
     open("scaler_y_minmax.pkl", "rb") as f_scaler_y:
    model = pickle.load(f_model)
    scaler_x = pickle.load(f_scaler_x)
    scaler_y = pickle.load(f_scaler_y)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_formatted = None
    if request.method == "POST":
        try:
            # Ambil data dari form
            luas_bangunan = float(request.form["luas_bangunan"])
            luas_tanah = float(request.form["luas_tanah"])
            jumlah_kamar = int(request.form["jumlah_kamar"])
            jumlah_toilet = int(request.form["jumlah_toilet"])
            luas_garasi = float(request.form["luas_garasi"])
            
            # Prediksi
            data_untuk_prediksi = np.array([[luas_bangunan, luas_tanah, jumlah_kamar, jumlah_toilet, luas_garasi]])
            data_untuk_prediksi_scaled = scaler_x.transform(data_untuk_prediksi)
            harga_prediksi_scaled = model.predict(data_untuk_prediksi_scaled)
            harga_prediksi_asli = scaler_y.inverse_transform(harga_prediksi_scaled.reshape(-1, 1))
            prediction = math.floor(harga_prediksi_asli[0][0])
            prediction_formatted = "{:,.0f}".format(prediction).replace(",", ".")
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction_formatted)

if __name__ == "__main__":
    app.run(debug=True)
