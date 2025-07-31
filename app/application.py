import os

from flask import Flask, render_template, request, session, redirect, url_for
from app.common.logger import get_logger
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(42)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters["nl2br"] = nl2br

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []
        
    if request.method == "POST":
        user_input = request.form.get("prompt")
        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages
            print(user_input)
            
            try:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    raise Exception("QA chain could not be created (LLM or VectorStore issue)")
                
                llm_input = {"query": user_input}
                response = qa_chain.invoke(llm_input)
                result = response.get("result" , "No response")
                
                messages.append({"role": "assistant", "content": result})
                
            except Exception as error:
                error_message = f"Error: {str(error)}"
                return render_template("index.html", messages=session["messages"], error=error_message)
            
        return redirect(url_for("index"))
    return render_template("index.html", messages=session.get("messages", []))    

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)