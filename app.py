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
        return jsonify({
            "error": "Azure Vision not configured. Set VISION_KEY and "
                     "VISION_ENDPOINT in your environment settings."
        }), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")

    # Appending details=Celebrities to the query parameters
    api_url = f"{VISION_ENDPOINT}/vision/v3.2/analyze"
    params = {
        "visualFeatures": "Categories,Faces,Brands",
        "details": "Celebrities"
    }
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}

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
        return jsonify({
            "error": f"Azure returned status {resp.status_code}", 
            "details": resp.text
        }), resp.status_code

    azure_data = resp.json()

    processed_response = {
        "faces": [],
        "brands": [],
        "landmarks": []
    }

    # Extract general facial features
    for face in azure_data.get("faces", []):
        processed_response["faces"].append({
            "gender": face.get("gender"),
            "age": face.get("age"),
            "box": {
                "left": face["faceRectangle"]["left"],
                "top": face["faceRectangle"]["top"],
                "width": face["faceRectangle"]["width"],
                "height": face["faceRectangle"]["height"]
            }
        })

    # Safely iterate through category details for identified celebrities
    for category in azure_data.get("categories", []):
        if "detail" in category and "celebrities" in category["detail"]:
            for celeb in category["detail"]["celebrities"]:
                celeb_name = celeb.get("name")
                celeb_box = celeb.get("faceRectangle")
                
                match_found = False
                if celeb_box:
                    for face_obj in processed_response["faces"]:
                        # Compare bounding boxes with a small variance buffer
                        if abs(face_obj["box"]["left"] - celeb_box["left"]) < 20:
                            # Fixed comment error and cleaned formatting assignment
                            face_obj["gender"] = celeb_name  
                            face_obj["age"] = f"Match: {round(celeb.get('confidence', 0) * 100)}%"
                            match_found = True
                            break
                
                # Append if it couldn't align cleanly with an existing face object
                if not match_found and celeb_box:
                    processed_response["faces"].append({
                        "gender": celeb_name,
                        "age": f"Match: {round(celeb.get('confidence', 0) * 100)}%",
                        "box": {
                            "left": celeb_box["left"],
                            "top": celeb_box["top"],
                            "width": celeb_box["width"],
                            "height": celeb_box["height"]
                        }
                    })

    # Extract corporate brands
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

    # Extract landmarks
    for category in azure_data.get("categories", []):
        if "detail" in category and "landmarks" in category["detail"]:
            for landmark in category["detail"]["landmarks"]:
                processed_response["landmarks"].append({
                    "name": landmark.get("name"),
                    "confidence": round(landmark.get("confidence", 0) * 100, 1)
                })

    return jsonify(processed_response), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": bool(VISION_KEY and VISION_ENDPOINT)})


app = app

if __name__ == "__main__":
    app.run(debug=True)
