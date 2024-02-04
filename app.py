from flask import Flask, request, jsonify
from analyse import analyseVideo

app = Flask(__name__)

@app.route('/get_genre', methods=['GET'])
def get_genre():
    path = request.args.get('path', '/path/to/your/video.mp4')
    
    # Use analyseVideo function
    analysis_results = analyseVideo(path)
    
    # Return the results as JSON
    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
