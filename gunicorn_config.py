from waitress import serve

from server import app  # Replace 'server' with your Flask app module name

serve(app, host='127.0.0.1', port=5000)