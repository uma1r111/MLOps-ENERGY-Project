from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>⚙️ Model Training Service</title>
        <style>
            body {
                background: linear-gradient(135deg, #2980b9, #6dd5fa, #ffffff);
                background-size: 200% 200%;
                animation: gradientMove 10s ease infinite;
                color: white;
                font-family: 'Poppins', sans-serif;
                margin: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                text-align: center;
            }

            @keyframes gradientMove {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            h1 {
                font-size: 2.5em;
                margin-bottom: 0.4em;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            }

            p {
                font-size: 1.2em;
                margin-bottom: 1.5em;
                opacity: 0.9;
            }

            .card {
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(8px);
                border-radius: 15px;
                padding: 25px 40px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
            }

            .card:hover {
                transform: scale(1.03);
            }

            footer {
                position: absolute;
                bottom: 20px;
                font-size: 0.9em;
                opacity: 0.8;
            }
        </style>
    </head>
    <body>
        <h1>⚙️ Model Training Service</h1>
        <div class="card">
            <p>Status: <strong>Running Smoothly ✅</strong></p>
        </div>
        <footer>© 2025 MLOps ENERGY Project</footer>
    </body>
    </html>
    """

@app.route("/health")
def health():
    # Allow both HTML and JSON modes
    if request.args.get("json") == "1":
        return jsonify(status="healthy", service="model_training")

    # Otherwise, show the styled health UI
    status = "healthy"
    color = "#00ffae" if status == "healthy" else "#ff6b6b"
    emoji = "✅" if status == "healthy" else "❌"

    return render_template_string(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>⚙️ Model Training Health</title>
        <style>
            body {{
                background: linear-gradient(135deg, #6dd5fa, #2980b9);
                background-size: 200% 200%;
                animation: gradientMove 10s ease infinite;
                color: white;
                font-family: 'Poppins', sans-serif;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                text-align: center;
            }}
            @keyframes gradientMove {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            .card {{
                background: rgba(255,255,255,0.1);
                padding: 30px 60px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            }}
            h1 {{ font-size: 2.3em; margin-bottom: 10px; }}
            .status {{ font-size: 1.5em; color: {color}; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>⚙️ Model Training Health</h1>
            <p class="status">{emoji} {status.title()}</p>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
