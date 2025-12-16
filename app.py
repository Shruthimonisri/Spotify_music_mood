# ==============================
# Mood Music AI - Streamlit App
# ==============================

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# Load Model, Scaler
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model('model/mood_model.h5')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

model, scaler = load_resources()

# ------------------------------
# Mood Mappings
# ------------------------------
moods = ['Sad', 'Happy', 'Energetic', 'Calm']
mood_emojis = {
    'Sad': '😢',
    'Happy': '😊',
    'Energetic': '🔥',
    'Calm': '😌'
}

# ------------------------------
# Playlist Suggestions
# ------------------------------
playlists = {
    'Sad': ['Someone Like You - Adele', 'Tears in Heaven - Eric Clapton', 'Hurt - Johnny Cash'],
    'Happy': ['Uptown Funk - Mark Ronson ft. Bruno Mars', 'Happy - Pharrell Williams', "Can't Stop the Feeling! - Justin Timberlake"],
    'Energetic': ['Thunderstruck - AC/DC', 'Eye of the Tiger - Survivor', 'We Will Rock You - Queen'],
    'Calm': ['Weightless - Marconi Union', 'River Flows in You - Yiruma', "Comptine d'un autre été - Yann Tiersen"]
}

# ------------------------------
# Feature Explanations
# ------------------------------
feature_explanations = {
    'danceability': 'How suitable for dancing (0-1). Higher values mean more danceable.',
    'energy': 'Perceived energy level (0-1). Higher for more energetic tracks.',
    'loudness': 'Overall loudness in dB (-60 to 0). Louder tracks have higher values.',
    'speechiness': 'Presence of spoken words (0-1). Higher for more speech-like.',
    'acousticness': 'Confidence of acoustic sound (0-1). Higher for acoustic tracks.',
    'instrumentalness': 'Likelihood of no vocals (0-1). Higher for instrumental tracks.',
    'liveness': 'Presence of live audience (0-1). Higher for live recordings.',
    'valence': 'Musical positiveness (0-1). Higher for happier, more positive tracks.',
    'tempo': 'Estimated tempo in BPM (0-250).'
}

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title='Mood Music AI', page_icon='🎵', layout='wide')

st.title('🎵 Mood Music AI')
st.markdown('Predict the mood of a song based on its audio features! Adjust the sliders and click predict.')

# Theme toggle (Streamlit has built-in dark mode)
# Note: Streamlit automatically supports dark/light mode based on user preference

# Input Sliders
col1, col2 = st.columns(2)

with col1:
    danceability = st.slider('Danceability', 0.0, 1.0, 0.5, help=feature_explanations['danceability'])
    energy = st.slider('Energy', 0.0, 1.0, 0.5, help=feature_explanations['energy'])
    loudness = st.slider('Loudness (dB)', -60.0, 0.0, -10.0, help=feature_explanations['loudness'])
    speechiness = st.slider('Speechiness', 0.0, 1.0, 0.1, help=feature_explanations['speechiness'])
    acousticness = st.slider('Acousticness', 0.0, 1.0, 0.2, help=feature_explanations['acousticness'])

with col2:
    instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0, help=feature_explanations['instrumentalness'])
    liveness = st.slider('Liveness', 0.0, 1.0, 0.1, help=feature_explanations['liveness'])
    valence = st.slider('Valence', 0.0, 1.0, 0.5, help=feature_explanations['valence'])
    tempo = st.slider('Tempo (BPM)', 0.0, 250.0, 120.0, help=feature_explanations['tempo'])

# Predict Button
if st.button('🔮 Predict Mood'):
    # Prepare features
    features = np.array([[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]])
    features_scaled = scaler.transform(features)
    
    # Predict
    pred = model.predict(features_scaled)
    pred_class = np.argmax(pred, axis=1)[0]
    mood = moods[pred_class]
    confidence = pred[0][pred_class] * 100
    
    # Display Result
    st.success(f'**Predicted Mood: {mood} {mood_emojis[mood]}**')
    st.info(f'**Confidence: {confidence:.1f}%**')
    
    # Probability Bar Chart
    st.subheader('Mood Probabilities')
    prob_dict = {moods[i]: pred[0][i] for i in range(len(moods))}
    st.bar_chart(prob_dict)
    
    # Playlist Suggestion
    st.subheader(f'🎶 Suggested {mood} Playlist')
    for song in playlists[mood]:
        st.write(f'• {song}')
    
    # Why this mood? (Feature Importance Explanation)
    st.subheader('Why this mood? Feature Analysis')
    st.markdown('Based on the feature values, here\'s a simple explanation:')
    if mood == 'Happy':
        st.write('- High valence and danceability often indicate happy, upbeat songs.')
    elif mood == 'Sad':
        st.write('- Lower valence and energy, combined with acousticness, suggest sad emotions.')
    elif mood == 'Energetic':
        st.write('- High energy, loudness, and tempo point to energetic, high-intensity tracks.')
    elif mood == 'Calm':
        st.write('- Lower energy, higher acousticness, and moderate tempo suggest calm, relaxing music.')
    
    # Show feature values
    st.subheader('Your Input Features')
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    for name, val in zip(feature_names, features[0]):
        st.write(f'**{name.capitalize()}:** {val:.3f}')

st.markdown('---')
st.markdown('Built with ❤️ using TensorFlow and Streamlit. A fun way to explore music moods!')