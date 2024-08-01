from flask import Flask, request, render_template_string, session, send_from_directory
import pickle
import numpy as np
import json
import os

# Inicializar la aplicación Flask
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necesario para usar sesiones

# Ruta base de la aplicación
base_path = os.path.dirname(__file__)

# Especificar la ruta a los archivos de datos
data_path = os.path.join(base_path, 'data')
app_path = base_path

# Cargar el modelo entrenado y otros archivos necesarios
model = pickle.load(open(os.path.join(data_path, 'chatbot_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(data_path, 'conversations_vectorizer_bow.pkl'), 'rb'))
with open(os.path.join(data_path, 'conversations_category_answers.json'), 'r', encoding='utf-8') as file:
    category_answers = json.load(file)

# Lista de categorías conocidas
known_categories = list(category_answers.keys())

# Preprocesar y vectorizar el texto de entrada
def preprocess_text(text):
    return vectorizer.transform([text]).toarray()

# Obtener respuesta basada en la categoría predicha
def get_response(predicted_category):
    if predicted_category in known_categories:
        return np.random.choice(category_answers[predicted_category])
    else:
        return "Lo siento, no puedo ayudarte con eso. ¿De qué signo quieres información?"

# Leer el contenido del archivo HTML
with open(os.path.join(app_path, 'index.html'), 'r', encoding='utf-8') as file:
    index_html_content = file.read()

# Ruta para servir archivos estáticos como la imagen de fondo
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app_path, filename)

# Ruta principal para la interfaz del chatbot
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Inicializar el historial de chat al cargar la página
        session['chat_history'] = [('Amira ChatBot', '¡Hola! Soy Amira ChatBot. ¿Cuál es tu nombre?')]
        session['user_name'] = None

    elif request.method == "POST":
        user_input = request.form["user_input"]

        # Verificar si el usuario ya ha proporcionado su nombre
        if session.get('user_name') is None:
            session['user_name'] = user_input
            session['chat_history'].append(('Usuario', user_input))
            session['chat_history'].append(('Amira ChatBot', f"Encantada de conocerte, {user_input}. ¿En qué puedo ayudarte hoy?"))
        else:
            session['chat_history'].append(('Usuario', user_input))
            processed_input = preprocess_text(user_input)
            predicted_category = model.predict(processed_input)[0]

            # Comprobación de categorías conocidas
            if predicted_category in known_categories:
                response = get_response(predicted_category)
            else:
                response = "Lo siento, no puedo ayudarte con eso. ¿De qué signo quieres información?"

            session['chat_history'].append(('Amira ChatBot', response))

        session.modified = True  # Asegura que la sesión se guarde correctamente

    chat_history = session.get('chat_history', [])
    return render_template_string(index_html_content, chat_history=chat_history)

# Ruta para verificar el estado de la aplicación
@app.route("/healthcheck")
def healthcheck():
    return "La aplicación está funcionando correctamente."

if __name__ == "__main__":
    app.run(debug=True)




