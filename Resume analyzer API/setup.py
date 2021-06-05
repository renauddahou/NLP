import pickle

import flask
import numpy as np
from flask import Flask, jsonify, request
from spacy.matcher import matcher
from tika import parser

from src.resume_data_prep import Feature_Matrix, Resume_Extractor, patterns

app=Flask(__name__)


def data_prep(resumelist):
    features,matcher=patterns()  #examples
    resume_obj=Resume_Extractor(resumelist)
    matches,doclist=resume_obj(matcher)
    # print(Resume_Extractor.doc_list)
    # for match_id,pos1,pos2 in matches: 
    #   print(f'{nlp.vocab.strings[match_id]} : {doc[pos1:pos2].text}')


    arr_obj=Feature_Matrix(len(resumelist),len(features))
    x_data,y_data=arr_obj.feature_gen(matches,doclist,features)

    return x_data,y_data

@app.route("/predict",methods=["POST","GET"])
def predict():
    response={}
    y_pred,class_names,scores=[],[],[]
    resumelist=['Resume data/My resume optional.pdf','Resume data/My Resume.pdf']
    # resumelist=str(input().split())
    x_data,y_data=data_prep(resumelist)
    with open('assets/normalizer.pkl','rb') as f:
        normalizer=pickle.load(f)

    x_data=normalizer.transform(x_data)
    
    with open('assets/encoder.pkl','rb') as f:
        encoder=pickle.load(f)

    with open('models/classification_model.pkl','rb') as f:
        class_model=pickle.load(f)

    with open('models/regression_model.pkl','rb') as f:
        reg_model=pickle.load(f)

    
    y_pred.append([class_model.predict(x_data),reg_model.predict(x_data)])
    print(y_pred[0][0])
    print(np.ravel(y_pred[0][1]))
    classes=y_pred[0][0]
    score_val=np.ravel(y_pred[0][1])


    for class_val,score in zip(classes,score_val):
        class_names.append(encoder.inverse_transform([class_val]))
        scores.append(score)

    for name,label,score in zip(resumelist,class_names,scores):
        response[name] = [str(label[0]),str(score*100)]


    return jsonify(response)

    

if __name__=='__main__':
    app.run(debug=True)



    





