import numpy as np
abc = np.array([556,213,312,212,543])
c = abc.std()
d= abc.var()




import nltk as nt
tokens = nt.word_tokenize('It is a chair. my name is Mohsin Aslam. I went thererere "ssas LOL  Pakistan:)"')
tagged = nt.pos_tag(tokens)
sentiment = nt.sent_tokenize('It is a chair. my name is yasir . I went thererere "ssas LOL :)"')
entities = nt.chunk.ne_chunk(tagged)





from textblob import TextBlob
text = "وقال متحدث باسم الحوثيين إن هذا الهجوم أعقبه هجوم آخر استهدف العاملين في مجال الطوارئ في أرحب، على بعد 40 كيلومترا (25 ميلا) من مدينة صنعاء."
from langdetect import detect
lang = detect(text)
import requests
url = 'http://translate.google.com/translate_a/t'
params = {
"text": text,
"sl": lang,
"tl": "en",
"client": "p"}
print(requests.get(url, params=params).content)



if(lang == 'en'):
    blob = TextBlob(text)
    sent = blob.sentiment
    print(sent)
else:
    import requests
    url = 'http://translate.google.com/translate_a/t'
    params = {
        "text": text,
        "sl": lang,
        "tl": "en",
        "client": "p"
    }
    print(requests.get(url, params=params).content)
    blob = TextBlob('great')
    sent = blob.sentiment
    print(sent)
#en_blob = TextBlob(u'Simple is better than complex.')


from langdetect import detect
# print(detect(text))

b1 = TextBlob("And now for something completely different.")

print(b1.parse())

