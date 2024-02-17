import requests
import json
import datetime

def send_post_request(input_text, length):
    url = 'http://127.0.0.1:5000/generate'
    data = {'input': input_text, 'length': length}
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {'status': 'error', 'message': 'Failed to generate music'}

# Example usage
if __name__ == "__main__":
    input_text = "[KEY: G\n COLLECTION: G(add2) F7sus(add3) A-9 F-9 G7(b13) F#7(b13)(b9) A13sus(9) G(addb6)/D Eb9(#11)(omit3) G7 E7(#9) Bb13(#11)(9) A-7 C/D Eb/F G7sus(add3) E7(b9) B-7(addb6) D13sus(b9) C#-9 D#7(#5) F9sus Bb-11(9)\n"
    length = "400"  # Adjust the length as needed
    time1 = datetime.datetime.now()
    result = send_post_request(input_text, length)
    time2 = datetime.datetime.now()
    diff = time2 - time1
    print (diff.total_seconds())
    print(result)
