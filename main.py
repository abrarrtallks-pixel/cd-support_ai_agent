from flask import Flask, request, jsonify
from agent_orchestrator import handle_user_message

app = Flask(__name__)

@app.route("/")
def home():
    return "Support AI Agent Running 🚀"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message required"}), 400

    result = handle_user_message(user_message)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
