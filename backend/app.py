from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from routes.auth_routes import auth
from routes.predict_routes import predict

load_dotenv()

app = Flask(__name__)
CORS(app)

# Register routes
app.register_blueprint(auth)
app.register_blueprint(predict)

@app.get("/")
def home():
    return {"msg": "CardioNova Backend Running"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
