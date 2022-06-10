from flask import Flask, request, jsonify
from flask_cors import CORS

from prepare_haystack_pipeline import combined_p, clean_text

# Define the app
app = Flask(__name__)
# Load configs
app.config.from_object('config')
# Set CORS policies
CORS(app)


@app.route("/query", methods=["GET"])
def qa():
	records = {'documents': []}

	if request.args.get("query"):
		query = request.args.get("query")

		records = combined_p.run(
			query = clean_text(query),
			params = {
				"BMRetriever": {"top_k": 3},
				"ESRetriever": {"top_k": 3}
			})
	else:
		return {"error": "Couldn't process your request"}, 422

	result = [{
		'question': r.meta['question'],
		'answers': r.meta['answers'],
		'score': r.score}
		for r in records['documents']]
	return jsonify(result)


if __name__ == '__main__':
	app.run(debug=True, host="0.0.0.0", port=5000)
