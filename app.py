from flask import Flask, render_template, request

try:
    from Python_BM25_search_engine import loadData, search
except:
    from RTA_Legislations_Chatbot.Flask_RTA_Legalaozation__BM25_Smart_Search_Engine.Python_BM25_search_engine import \
        loadData, search

app = Flask(__name__)
tok_key, tok_text = loadData()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    results = search(userText, tok_key, tok_text)
    a = ' '.join(str(results[0]).split())
    return results[0][0] + '||||||' + results[0][1] + \
           "-----------------------------------------------------------------" + \
           results[1][0] + '||||||' + results[1][1] + \
           "-----------------------------------------------------------------" + \
           results[2][0] + '||||||' + results[2][1]


if __name__ == "__main__":
    app.run()
