from PIL import Image
import sys
import codecs
import pyocr
# import pyocr.builders

tools = pyocr.get_available_tools()
if len(tools) == 0:
 print("No OCR tool found")
 sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
lang = langs[0]



builder = pyocr.builders.DigitBuilder(tesseract_layout=3)

txt = tool.image_to_string(
 Image.open(r"C:\01_works\Dev\Jupyter\bot_analyzation\bot_ocr\iCloud Photos\IMG_2980.PNGt.png"),
 lang=lang,
 builder=builder
)

if len(txt) != 0:
    print(txt)
else:
    print("no text detected.")

# for t in txt:
#     print(t.content)


with codecs.open("toto.txt", 'w', encoding='utf-8') as file_descriptor:
    builder.write_file(file_descriptor, txt)
