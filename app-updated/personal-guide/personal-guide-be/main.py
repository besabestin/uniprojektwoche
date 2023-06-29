import os
import json
import openai
from flask import Flask, request, jsonify

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


@app.route('/')
def hello_world():
    print('running completion')
    print(openai.api_key)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Who was the first president of the US?"}
        ]
    )
    print(completion)
    #return f'Hello World!{completion["choices"][0]["message"]["content"]}'
    response = jsonify(completion)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@app.route('/chat', methods=["POST", "OPTIONS"])
def respond_to_chat():
    """
    api endpoint to respond to chatbot requests
    returns completion coming from openai api
    """
    
    if request.method == 'POST':
        reqbody = request.json
        ret = ""
        if 'messages' in reqbody and len(reqbody['messages']) > 0:
            lastprompt = reqbody['messages'][-1]
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"reply to the following prompt as if you are an annoying chatbot, reply annoyed: {lastprompt}"}
                ]
            )
            ret = completion["choices"][0]["message"]["content"]
        return corsify_response(jsonify({"message": ret}))
    
    return corsify_response(jsonify({"message": "options"}))


def get_story_stats(the_story):
    main_events = ["event1", "event2"]
    general_sentiment = "Happy"
    return (main_events, general_sentiment)


def get_sentiment(content):
    #connects to openai api to get sentiment
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user", 
                "content": f"""
                    Get the sentiment out of this diary content using a single 
                    humanly word like happy, sad, exciting, annoying etc.: {content}"""
            }
        ]
    )
    ret = completion["choices"][0]["message"]["content"]
    return ret


def get_events(content):
    #connects to openai api to extract main events of content
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user", 
                "content": f"""
                Summarize this diary content by extracting two representative events out.
                Reply as if you are an annoying chatbot. 
                Use maximum 3 words per event and separate the events using | : {content}"""
            }
        ]
    )
    ret = completion["choices"][0]["message"]["content"]
    return ret


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
        content = ""
        if reqbody['diary'] and len(reqbody['diary']) > 0:
            for diaryentry in reqbody['diary']:
                content += f"{diaryentry['when']} {diaryentry['content']}"
            prompt = reqbody['entry']
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user", 
                        "content": f"""
                        I ll give you diary entries for several days and 
                        you will answer question based on the diary entries. 
                        I ll mark the start of the question with the word Qn. 
                        The diary entries will be date entries followed by text. 
                        {content} Qn: {prompt}"""
                    }
                ]
            )
            ret = completion["choices"][0]["message"]["content"]
            return corsify_response(jsonify({"message": ret}))
    return corsify_response(jsonify({"message": "options"}))

@app.route('/visualizer', methods=["POST", "OPTIONS"])
def get_images():
    if request.method == 'POST':
        reqbody = request.json
        prompts = reqbody["content"].split("|")
        images = []
        for entry in prompts:
            if entry and len(entry) > 0:
                response = openai.Image.create(
                    prompt = entry,
                    n=1,
                    size="256x256"
                )
                image_url = response['data'][0]['url']
                images.append(image_url)
        print(images)
        return corsify_response(jsonify({"images": images}))

    return corsify_response(jsonify({"message": "options"}))
        

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5153)

# in mac
# echo 'export OPENAI_API_KEY=<>' >> ~/.zshrc'
# source ~/.zshrc