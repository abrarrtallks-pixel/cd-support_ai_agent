from flask import Flask, request, jsonify
from agent_orchestrator import handle_user_message

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Support Bot Running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")

    response = handle_user_message(user_message)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
