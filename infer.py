import onnxruntime as ort
import numpy as np
import os

path = os.path.join("models" ,"model.onnx")
session = ort.InferenceSession(path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict_survival(features):
    input_array = np.array(features 
                           ,dtype=np.float32).reshape(1,-1)
    output = session.run([output_name] ,
                         {input_name:input_array})[0]
    probability = float(output[0][0])
    prediction =  int(probability>0.5)
    return prediction

if __name__ == '__main__':
    features = [.2 ,.3 ,.3 ,.5 ,.4 ,.8 ,.2]
    prediction = predict_survival(features)
    print(prediction)