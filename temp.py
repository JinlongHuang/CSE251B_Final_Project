import googletrans
import time

translator = googletrans.Translator(service_urls=['translate.googleapis.com','translate.google.com','translate.google.co.kr'])
lang = translator.detect("ہیگ کی تفتیش ایف بی آئی اہلکاروں  کی طرف سے کی گئی")
time.sleep(1)
result = translator.translate("ہیگ کی تفتیش ایف بی آئی اہلکاروں  کی طرف سے کی گئی", src='ur', dest='en')
print(lang)
print(result.text)