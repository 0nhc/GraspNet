from flask import Flask, request, jsonify


class GSNetFlaskServer:
    def __init__(self):
        self.app = Flask(__name__)
        self._init = False
        self.pcd_ros = []

        self.setup_routes()
    
    
    # Method to set up routes
    def setup_routes(self):
        @self.app.route('/get_gsnet_grasp', methods=['POST'])
        def get_gsnet_grasp():
            data = request.json  # Get JSON data from the request
            return jsonify(data)  # Return the data back as a JSON response
    

    def run(self):
        self.app.run(debug=True)


gsnetflaskserver = GSNetFlaskServer()
gsnetflaskserver.run()