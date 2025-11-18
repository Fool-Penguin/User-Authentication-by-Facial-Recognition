# server.py
from flask import Flask, request, jsonify, send_from_directory, session, abort
import os
from face_service import process_image_bytes

app = Flask(__name__)
# set a secret key for session cookies (replace in production)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_change_me")

# Serve main webcam page
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# Success page after authorized login (POST only)
@app.route("/success", methods=["POST"])
def success():
    if not session.get("authorized"):
        return abort(403)
    return send_from_directory(".", "success.html")

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

    if result.get("ok") and result.get("recognized") and result.get("real") and result.get("authorized"):
        session["authorized"] = True
        session["name"] = result.get("name")
    else:
        session.pop("authorized", None)
        session.pop("name", None)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
