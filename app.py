from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

# --- Configuración de Flask ---
app = Flask(__name__)

# --- Configuración de Spotify ---
SPOTIFY_CLIENT_ID = '4b14bcee621141b090bc8402f862dd42'
SPOTIFY_CLIENT_SECRET = '89398db9ea9a44908e25575085fbdcc9'
SPOTIFY_REDIRECT_URI = 'http://127.0.0.1:8888/callback'

# --- Paths ---
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "dataset", "spotify_tracks2.csv")
model_path = os.path.join(base_path, "model", "recommender.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# --- Cargar y limpiar dataset ---
df = pd.read_csv(
    csv_path,
    header=0,
    names=['name', 'artist_name', 'genre', 'uri', 'danceability', 'energy', 'valence', 'tempo']
)

# Eliminar filas vacías o con espacios entre géneros
df = df.dropna(how='any')
df = df[~df['name'].astype(str).str.strip().eq('')]

# Normalizar texto
df['genre'] = df['genre'].astype(str).str.lower().str.strip()
df['name'] = df['name'].astype(str).str.strip()
df['artist_name'] = df['artist_name'].astype(str).str.strip()

# Convertir numéricas
num_cols = ['danceability', 'energy', 'valence', 'tempo']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores numéricos inválidos
df = df.dropna(subset=num_cols)

# Features
features = ['danceability', 'energy', 'valence', 'tempo']

print("✅ Dataset cargado correctamente")
print(f"🎵 Total canciones: {len(df)}")
print(f"🎶 Géneros disponibles: {df['genre'].unique()}")

# --- Modelo global de respaldo ---
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = NearestNeighbors(n_neighbors=10, metric='cosine')
    model.fit(df[features])
    joblib.dump(model, model_path)

# --- Recomendación ---
def recomendar_playlist_por_genero(generos):
    generos = [g.lower().strip() for g in generos]
    df_genero = df[df['genre'].isin(generos)]

    if df_genero.empty:
        return {"error": f"No se encontraron canciones para: {', '.join(generos)}"}

    # Si hay pocas canciones, devuélvelas todas
    if len(df_genero) <= 10:
        canciones = [{
            "name": row['name'],
            "artist": row['artist_name'],
            "uri": row['uri']
        } for _, row in df_genero.iterrows()]

        return {
            "base": {
                "name": df_genero.iloc[0]['name'],
                "artist": df_genero.iloc[0]['artist_name']
            },
            "canciones": canciones
        }

    # Elegir canción base aleatoria
    base = df_genero.sample(1, random_state=np.random.randint(0, 10000))

    # Modelo local solo con ese género
    modelo_local = NearestNeighbors(n_neighbors=min(11, len(df_genero)), metric='cosine')
    modelo_local.fit(df_genero[features])

    dist, indices = modelo_local.kneighbors(base[features])
    indices = indices[0][1:11]  # excluir la base

    recomendadas = df_genero.iloc[indices]

    canciones = [{
        "name": row['name'],
        "artist": row['artist_name'],
        "uri": row['uri']
    } for _, row in recomendadas.iterrows()]

    return {
        "base": {
            "name": base.iloc[0]['name'],
            "artist": base.iloc[0]['artist_name']
        },
        "canciones": canciones
    }

# --- Crear playlist en Spotify ---
def crear_playlist_en_spotify(nombre, track_uris):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="playlist-modify-public,playlist-modify-private,user-library-read"
    ))
    user_id = sp.me()["id"]
    playlist = sp.user_playlist_create(user=user_id, name=nombre, public=True)
    sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)
    return playlist["external_urls"]["spotify"]

# --- Endpoint Flask ---
@app.route("/recomendar", methods=["POST"])
def recomendar():
    data = request.get_json()
    genero = data.get("genero")

    if not genero:
        return jsonify({"error": "Debe proporcionar un género"}), 400

    resultado = recomendar_playlist_por_genero([genero])
    if "error" in resultado:
        return jsonify(resultado), 404

    uris = [c["uri"] for c in resultado["canciones"]]

    try:
        url_playlist = crear_playlist_en_spotify(f"Playlist IA - {genero.capitalize()}", uris)
        resultado["playlist_url"] = url_playlist
    except Exception as e:
        resultado["playlist_url"] = f"Error al crear playlist: {str(e)}"

    return jsonify(resultado)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


