# app.py
from flask import Flask, render_template, request
from final import summarize

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("indexhome.html")

@app.route("/index1.html",methods=["GET", "POST"])
def first_page():
    if request.method == "POST":
        data = request.form["data"]

        # Use the summarize function from final.py
        summary = summarize(data)

        return render_template("index1.html", result=summary)
    else:
        return render_template("index1.html")


@app.route("/index2.html")
def second_page():
    return render_template("index2.html")

if __name__ == "__main__":
    app.run(debug=True)