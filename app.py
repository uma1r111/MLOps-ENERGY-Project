from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLOps ENERGY Dashboard</title>
        <style>
            body {
                background: linear-gradient(135deg, #1f4037, #99f2c8);
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
                font-size: 2.8em;
                margin-bottom: 0.4em;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            }

            p {
                font-size: 1.2em;
                margin-bottom: 2em;
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

            .status {
                font-size: 1.5em;
                font-weight: bold;
                margin-top: 10px;
            }

            .status.ok {
                color: #00ffae;
            }

            .status.down {
                color: #ff6b6b;
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
        <h1>⚡ MLOps ENERGY Dashboard</h1>
        <p>Monitoring Server's health in real-time</p>

        <div class="card">
            <div id="status" class="status">Checking...</div>
        </div>

        <footer>© 2025 MLOps ENERGY Project</footer>

        <script>
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    const statusDiv = document.getElementById('status');

                    if (data.status === 'ok') {
                        statusDiv.textContent = '✅ Server is Healthy';
                        statusDiv.className = 'status ok';
                    } else {
                        statusDiv.textContent = '⚠️ Server Issue Detected';
                        statusDiv.className = 'status down';
                    }
                } catch (error) {
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = '❌ Server Unreachable';
                    statusDiv.className = 'status down';
                }
            }

            // Run every 5 seconds
            checkHealth();
            setInterval(checkHealth, 5000);
        </script>
    </body>
    </html>
    """, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
