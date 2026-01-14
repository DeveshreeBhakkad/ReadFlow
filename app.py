from flask import Flask, render_template, request
from recommender import rank_blogs
from search import search_blogs   # your web search file

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    grouped_results = None

    if request.method == "POST":
        query = request.form.get("query")

        blogs = search_blogs(query)
        ranked = rank_blogs(query, blogs)

        grouped_results = {
            "Beginner": [],
            "Intermediate": [],
            "Advanced": []
        }

        for blog in ranked:
            grouped_results[blog["level"]].append(blog)

    return render_template("index.html", results=grouped_results)

if __name__ == "__main__":
    app.run(debug=True)

