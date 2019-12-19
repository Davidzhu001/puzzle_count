import cv2
import numpy as np
from flask import Flask, json, request

app = Flask(__name__)


@app.route("/count", methods=['POST', 'GET'])

def count():
    if request.method == 'POST':
        if 'org' not in request.files:
            return json.dumps({"message": 'No image of original picture', "success": False})
        org = request.files['org']
        # idfront = Image.open(request.files['idfront'].stream)
        org.save('original_image.jpg')

        if 'det' not in request.files:
            return json.dumps({"message": 'No image of right arrow', "success": False})
        det = request.files['det']
        # det = Image.open(request.files['det'].stream)
        det.save('predict_image.jpg')

    img_rgb = cv2.imread('original_image.jpg')     #main image

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('predict_image.jpg', 0)  #object image

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    f = set()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,0), 2)
        sensitivity = 100
        f.add((round(pt[0]/sensitivity), round(pt[1]/sensitivity)))

    #cv2.imwrite('resulting_image.png', img_rgb)   #to save the resulting image with rectangular boxes

    found_count = int(len(f))
    a = {"count": found_count}      #It will save a 'count.json' file with all the data
    with open('count.json', 'w') as json_file:
        json.dump(a, json_file)
    return json.dumps({"count": found_count})

@app.route("/")
def hello():
    return "Counting specific object in image"


if __name__ == "__main__":
    app.run(debug=True)
