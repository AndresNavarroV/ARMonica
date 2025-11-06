from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import spotipy
import os
import json
import time
import requests

# --- Configuración de Flask ---
app = Flask(__name__)

# --- Configuración de Spotify (Render) ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SPOTIFY_CACHE_JSON = os.getenv("SPOTIFY_CACHE_JSON")

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

df = df.dropna(how='any')
df = df[~df['name'].astype(str).str.strip().eq('')]
df['genre'] = df['genre'].astype(str).str.lower().str.strip()
df['name'] = df['name'].astype(str).str.strip()
df['artist_name'] = df['artist_name'].astype(str).str.strip()

num_cols = ['danceability', 'energy', 'valence', 'tempo']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=num_cols)

features = ['danceability', 'energy', 'valence', 'tempo']

print("✅ Dataset cargado correctamente")
print(f"🎵 Total canciones: {len(df)}")
print(f"🎶 Géneros disponibles: {df['genre'].unique()}")

# --- Modelo global ---
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = NearestNeighbors(n_neighbors=10, metric='cosine')
    model.fit(df[features])
    joblib.dump(model, model_path)

# --- Refrescador automático de token ---
def get_spotify_token():
    """
    Refresca automáticamente el access_token usando el refresh_token de SPOTIFY_CACHE_JSON.
    """
    if not SPOTIFY_CACHE_JSON:
        raise Exception("No se encontró la variable SPOTIFY_CACHE_JSON.")

    spotify_cache = json.loads(SPOTIFY_CACHE_JSON)
    now = int(time.time())

    if now >= spotify_cache.get("expires_at", 0):
        print("🔄 Token expirado. Refrescando...")

        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": spotify_cache["refresh_token"],
                "client_id": SPOTIFY_CLIENT_ID,
                "client_secret": SPOTIFY_CLIENT_SECRET,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Error al refrescar token: {response.text}")

        new_data = response.json()
        spotify_cache["access_token"] = new_data["access_token"]
        spotify_cache["expires_at"] = now + new_data.get("expires_in", 3600)

        os.environ["SPOTIFY_CACHE_JSON"] = json.dumps(spotify_cache)
        print("✅ Token actualizado correctamente.")

    return spotify_cache["access_token"]

# --- Recomendación de canciones ---
def recomendar_playlist_por_genero(generos):
    generos = [g.lower().strip() for g in generos]
    df_genero = df[df['genre'].isin(generos)]

    if df_genero.empty:
        return {"error": f"No se encontraron canciones para: {', '.join(generos)}"}

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

    base = df_genero.sample(1, random_state=np.random.randint(0, 10000))
    modelo_local = NearestNeighbors(n_neighbors=min(11, len(df_genero)), metric='cosine')
    modelo_local.fit(df_genero[features])
    dist, indices = modelo_local.kneighbors(base[features])
    indices = indices[0][1:11]
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

# --- Crear playlist pública en Spotify ---
def crear_playlist_en_spotify(nombre, track_uris):
    try:
        token = get_spotify_token()
        sp = spotipy.Spotify(auth=token)

        user_id = sp.me()["id"]
        playlist = sp.user_playlist_create(user=user_id, name=nombre, public=True)
        sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)

        return playlist["external_urls"]["spotify"]

    except Exception as e:
        return f"Error al crear playlist: {str(e)}"

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
    resultado["playlist_url"] = crear_playlist_en_spotify(f"Playlist IA - {genero.capitalize()}", uris)

    return jsonify(resultado)

# --- Servidor principal ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
