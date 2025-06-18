from flask import Flask
from flask_cors import CORS
import logging
import os

from embeddings.candidate_embeddings import candidates_bp

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register blueprints
app.register_blueprint(candidates_bp, url_prefix='/candidates')

@app.route('/')
def health_check():
    return {
        'status': 'healthy',
        'service': 'Candidate Search API',
        'version': '1.0'
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)