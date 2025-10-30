from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/")
def index():
    return "MLOps ENERGY Project is running!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)