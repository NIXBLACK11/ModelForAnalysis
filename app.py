from flask import Flask, request, jsonify
from audio import analyse

app = Flask(__name__)

@app.route('/analyse_audio', methods=['GET'])
def get_genre():
    path = request.args.get('path', '/path/to/your/video.mp4')
    # Use analyseVideo function
    analysis_results = analyse.analyse.analyse_audio(path)
    # Return the results as JSON
    return jsonify(analysis_results)


if __name__ == '__main__':
    app.run(debug=True)
