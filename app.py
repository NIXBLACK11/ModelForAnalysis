from flask import Flask, request, jsonify
from audio import analyse as aa
from video import analyse as va

app = Flask(__name__)

@app.route('/analyse_audio', methods=['GET'])
def get_audio_genre():
    path = request.args.get('path', '/path/to/your/video.mp4')
    analysis_results = aa.analyse_audio(path)
    return jsonify(analysis_results)


@app.route('/analyse_video', methods=['GET'])
def get_genre():
    path = request.args.get('path', '/path/to/your/video.mp4')
    analysis_results = va.analyse_video(path)
    return jsonify(analysis_results)


if __name__ == '__main__':
    app.run(debug=True)
