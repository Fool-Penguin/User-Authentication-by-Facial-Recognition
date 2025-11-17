# server.py
from flask import Flask, request, jsonify, send_from_directory
from face_service import process_image_bytes

app = Flask(__name__)

# Serve main webcam page
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# Success page after authorized login
@app.route("/success")
def success():
    return send_from_directory(".", "success.html")

# API endpoint for face recognition
@app.route("/recognize_face", methods=["POST"])
def recognize_face_route():
    file = request.files.get("image")
    if not file:
        return jsonify({
            "ok": False,
            "recognized": False,
            "real": None,
            "authorized": False,
            "name": None,
            "reason": "no_image",
            "time_ms": 0,
        }), 400

    image_bytes = file.read()
    result = process_image_bytes(image_bytes, database="faceDB")
    return jsonify(result)


if __name__ == "__main__":
    # Open http://localhost:5000 in your browser
    app.run(host="127.0.0.1", port=5000, debug=True)
