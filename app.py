import streamlit as st
from function import InputData, predict, model

st.title('Prédiction de Streams en fonction des attributs de la musique')

# Créer des sliders pour chaque attribut
danceability = st.slider('Danceability', 0.0, 100.0)
valence = st.slider('Valence', 0.0, 100.0)
energy = st.slider('Energy', 0.0, 100.0)
acousticness = st.slider('Acousticness', 0.0, 100.0)
instrumentalness = st.slider('Instrumentalness', 0.0, 100.0)
liveness = st.slider('Liveness', 0.0, 100.0)
speechiness = st.slider('Speechiness', 0.0, 100.0)

# Convert the values to dictionary with correct keys
input_dict = {
    'danceability_%': danceability,
    'valence_%': valence,
    'energy_%': energy,
    'acousticness_%': acousticness,
    'instrumentalness_%': instrumentalness,
    'liveness_%': liveness,
    'speechiness_%': speechiness,
}

# Créer un bouton pour faire la prédiction
if st.button('Prédire'):
    input_data = InputData(**input_dict)
    prediction = predict(input_data)
    st.write('Prédiction de Streams :', prediction['Streams prediction'])



# Section pour parler avec le modèle GPT-2
st.header('Parler avec le modèle GPT-2')
message = st.text_input('Entrez votre message ici')
if st.button('Envoyer le message'):
    response = model(message)
    st.markdown(f"<p style='color:orange;'>{response[0]['generated_text']}</p>", unsafe_allow_html=True)