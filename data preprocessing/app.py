from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>Data Preprocessing</title>
            <style>
                body { background-color: #f7f8fa; font-family: 'Segoe UI', sans-serif; text-align: center; padding-top: 100px; }
                .container { background: white; padding: 40px; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); display: inline-block; }
                h1 { color: #2563eb; font-size: 28px; margin-bottom: 10px; }
                p { color: #333; font-size: 18px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧩 Data Preprocessing Service</h1>
                <p>Status: <strong>Running Smoothly ✅</strong></p>
            </div>
        </body>
    </html>
    """

@app.route("/health")
def health():
    return jsonify(status="healthy", service="data_preprocessing")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8004)
