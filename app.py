import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

VISION_KEY = os.environ.get("VISION_KEY", "")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "").rstrip("/")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({"error": "Azure Vision not configured."}), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    # Clean endpoint construction
    api_url = f"{VISION_ENDPOINT}/vision/v3.2/analyze"
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    
    # Safe general features that work perfectly on Free/Standard tiers globally
    params = {
        "visualFeatures": "Categories,Description,Objects,Brands"
    }

    try:
        if image_url:
            headers["Content-Type"] = "application/json"
            resp = requests.post(api_url, headers=headers, params=params, json={"url": image_url}, timeout=20)
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception:
                return jsonify({"error": "Invalid base64 image data string"}), 400
                
            headers["Content-Type"] = "application/octet-stream"
            resp = requests.post(api_url, headers=headers, params=params, data=image_bytes, timeout=20)
        else:
            return jsonify({"error": "No image URL or image data provided"}), 400
    except requests.RequestException as exc:
        return jsonify({"error": f"Request to Azure AI Vision failed: {exc}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"Azure returned status {resp.status_code}", "details": resp.text}), resp.status_code

    azure_data = resp.json()

    # Create a safe, standardized output format for the frontend canvas
    processed_response = {
        "faces": [],
        "brands": [],
        "landmarks": []
    }

    # Extract descriptions/tags safely and format them as readable items
    description_obj = azure_data.get("description", {})
    captions = description_obj.get("captions", [])
    text_summary = captions[0].get("text", "Object detected") if captions else "Image analysis complete"

    # Map detected structural objects into the frontend canvas structure
    for obj in azure_data.get("objects", []):
        processed_response["faces"].append({
            "gender": obj.get("object", "Object"),
            "age": f"Confidence: {round(obj.get('confidence', 0) * 100)}%",
            "box": {
                "left": obj["rectangle"]["x"],
                "top": obj["rectangle"]["y"],
                "width": obj["rectangle"]["w"],
                "height": obj["rectangle"]["h"]
            }
        })

    for brand in azure_data.get("brands", []):
        processed_response["brands"].append({
            "name": brand.get("name"),
            "confidence": round(brand.get("confidence", 0) * 100, 1),
            "box": {
                "left": brand["rectangle"]["left"],
                "top": brand["rectangle"]["top"],
                "width": brand["rectangle"]["width"],
                "height": brand["rectangle"]["height"]
            }
        })

    return jsonify(processed_response), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": bool(VISION_KEY and VISION_ENDPOINT)})

app = app

if __name__ == "__main__":
    app.run(debug=True)
