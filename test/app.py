from flask import Flask, request, jsonify
from middle import callAnalyse

app = Flask(__name__)

@app.route('/get_genre', methods=['GET'])
def get_genre():
    path = request.args.get('path', '/path/to/your/video.mp4')
    videoGenre = request.args.get('genre', 'MrBeast')
    # Use analyseVideo function
    analysis_results = callAnalyse(path, videoGenre)
    
    # Return the results as JSON
    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
