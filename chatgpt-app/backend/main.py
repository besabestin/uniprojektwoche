import os
import json
import openai
from flask import Flask, request, jsonify

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


@app.route('/')
def hello_world():
    response = jsonify({"message": "hello!"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route('/chat', methods=["POST", "OPTIONS"])
def respond_to_chat():
    print("post request called")
    
    if request.method == 'POST':
        #reqbody = request.json
        return corsify_response(jsonify({"message": "hello again!"}))
    
    return corsify_response(jsonify({"message": "options"}))


def get_story_stats(the_story):
    main_events = ["event1", "event2"]
    general_sentiment = "Happy"
    return (main_events, general_sentiment)


def get_sentiment(content):
    #connects to openai api to get sentiment
    return "Happy"


def get_events(content):
    #connects to openai api to extract main events of content
    return "getting drunk"


def generate_story_stat(story):
    sentiment = get_sentiment(story["content"])
    events = get_events(story["content"])
    return {
        "sentiment": sentiment,
        "events": events,
        "when": story["when"],
        "content": story["content"]
    }


def generate_story_stats(stories):
    return [generate_story_stat(x) for x in stories]


@app.route('/newstory', methods=["POST", "OPTIONS"])
def add_story():
    if request.method == 'POST':
        print(request.json)
        reqbody = request.json
        story = {
            "when": reqbody["when"],
            "content": reqbody["content"]
        }
        story = generate_story_stat(reqbody)
        
        return corsify_response(jsonify(story))
    
    return corsify_response(jsonify({"message": "options"}))


@app.route('/stories')
def get_stories():
    stories = []
    # may be this comes from json
    print("checking this")
    stories = []
    with open('./stories.json', 'r') as f:
        stories = json.load(f)
    return corsify_response(jsonify({"stories": generate_story_stats(stories)}))


@app.route('/searchjournal', methods=["POST", "OPTIONS"])
def search_in_journal():
    if request.method == 'POST':
        reqbody = request.json
        print(reqbody)
        return corsify_response(jsonify({"message": "placeholder"}))
    return corsify_response(jsonify({"message": "options"}))


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5152)

# in mac
# echo 'export OPENAI_API_KEY=<>' >> ~/.zshrc'
# source ~/.zshrc