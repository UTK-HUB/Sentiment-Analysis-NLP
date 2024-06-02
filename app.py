from flask import Flask, jsonify, request
import pickle
from data_preprocessing import remove_lines,expand_text,Handle_accented_chars,remove_stopwords,tokenizer,lemmmatize_data,clean_data,joining_words

app = Flask(__name__)

countVec = pickle.load(open('countVec.pkl','rb'))
model = pickle.load(open('model_mnb.pkl','rb'))

@app.route('/')
def home():
    return jsonify({'response' : 'Welcome! To Sentiment Analysis.'})

@app.route('/predict',methods=['POST'])
def predict():
    requestdata = request.get_data(as_text=True)
    clean_test = remove_lines(requestdata)
    clean_test = expand_text(clean_test)
    clean_test = Handle_accented_chars(clean_test)
    clean_test = tokenizer(clean_test)
    clean_test = remove_stopwords(clean_test)
    clean_test = clean_data(clean_test)
    clean_test = lemmmatize_data(clean_test)
    clean_test = joining_words(clean_test)
    vector = countVec.transform([clean_test])
    prediction = model.predict(vector)
    if prediction[0]==0:
        result = 'Negative sentiment'
    elif prediction[0]==1:
        result = 'Neutral sentiment'
    elif prediction[0]==2:
        result = 'Positive sentiment'

    return jsonify({'Product Review' : requestdata, 'Result' : result})

if __name__ == '__main__':
    app.run(port=8080)