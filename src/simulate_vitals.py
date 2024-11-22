from flask import Flask, jsonify
import random

app = Flask(__name__)

def simulate_heartbeat():
    heart_rate = random.randint(60, 120)
    return heart_rate

def simulate_blood_pressure():
    systolic = random.randint(90, 180)
    diastolic = random.randint(60, 120)
    return systolic, diastolic

@app.route('/vitals', methods=['GET'])
def get_vitals():
    heartbeat = simulate_heartbeat()
    systolic, diastolic = simulate_blood_pressure()
    return jsonify({
        'heartbeat': heartbeat,
        'blood_pressure': {
            'systolic': systolic,
            'diastolic': diastolic
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
