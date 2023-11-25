from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/parse_html', methods=['POST'])
def parse_html():
    data = request.get_json()
    html_content = data.get('html', '')
    print(html_content)

    # Process HTML content (replace this with your logic)
    parsed_data = {'status': 'success', 'message': 'HTML processed successfully'}

    return jsonify(parsed_data)

if __name__ == '__main__':
    app.run(port=5000)