from PIL import Image
from pytesseract import pytesseract
import spacy
import cv2
import numpy as np
from spacy.language import Language
from utils.text_processing import process_text, is_ingredient_sent
import json

# set to True if you want to manually classify text
USER_INPUT = False

### --------------------- ###
#       Load Models         #
### --------------------- ###
topic_model = spacy.load("output/model-best")

### --------------------- ###
#       Image Import        #
### --------------------- ###

# Defining paths to tesseract.exe
# and the image we would be using
path_to_tesseract = r"/opt/homebrew/opt/tesseract/bin/tesseract"
image_path = r"sample_data/Tahini-test_kitchen.png"

# Opening the image & storing it in an image object
img = cv2.imread(image_path)

### --------------------- ###
#      Image Processing     #
### --------------------- ###

# --- dilation on the green channel ---
dilated_img = cv2.dilate(img[:, :, 1], np.ones((7, 7), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)

# --- finding absolute difference to preserve edges ---
diff_img = 255 - cv2.absdiff(img[:, :, 1], bg_img)

# --- normalizing between 0 to 255 ---
norm_img = cv2.normalize(
    diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
)
cv2.imshow("norm_img", cv2.resize(norm_img, (0, 0), fx=0.5, fy=0.5))

# --- Otsu threshold ---
th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("th", cv2.resize(th, (0, 0), fx=0.5, fy=0.5))


### --------------------- ###
#       Image to Text       #
### --------------------- ###

# # Providing the tesseract executable
# # location to pytesseract library
pytesseract.tesseract_cmd = path_to_tesseract

# # Passing the image object to image_to_string() function
# # This function will extract the text from the image
text = pytesseract.image_to_string(th)


### --------------------- ###
#         Read Text         #
### --------------------- ###

doc = process_text(text)


### --------------------- ###
#         Analyse Text      #
### --------------------- ###

# Find paragraphs by splitting on "whitespace" sentences
paragraphs = []
paragraph = []
for sent in doc.sents:
    if len(sent) <= 1:
        paragraphs.append(paragraph)
        paragraph = []
        continue
    paragraph.append(sent)
paragraphs.append(paragraph)

# Merge paragraphs into a single string
texts = []
for paragraph in paragraphs:
    text = ""
    if len(paragraph) > 1:
        text = " ".join([sent.text for sent in paragraph])
    else:
        text = paragraph[0].text

    # remove \n
    text = text.replace("\n", " ")
    texts.append(text)

### --------------------- ###
#         Get topics        #
### --------------------- ###
cats = []
for text in texts:
    doc = topic_model(text)
    potential_cats = doc.cats
    # get max value
    max_value = max(potential_cats.values())
    if max_value < 0.5 and USER_INPUT:
        # get input from user
        max_key = input(
            f"Could not classify {text}. \nPlease enter a category: ('title', 'blurb', 'ingredient', 'method', 'other')\nInput topic: "
        )
    else:
        # get key of max value
        max_key = [k for k, v in potential_cats.items() if v == max_value]
    cats.append({"text": text, "topic": max_key, "scores": potential_cats})

print(json.dumps(cats, indent=4))

# def get_serving_sentences(paragraphs: list) -> tuple[int, str]:
#     for idx, paragraph in enumerate(paragraphs):
#         if len(paragraph) > 1:
#             continue
#         if "SERVES" in paragraph[0].text:
#             return idx, paragraph[0]


# def get_serving_amount(serving_sentence) -> str:
#     for token in serving_sentence:
#         if token.pos_ == "NUM":
#             return token


# serving_index, serving_sentence = get_serving_sentences(paragraphs)
# serving_amount = get_serving_amount(serving_sentence)


# ingredients_indexes = [idx for idx, p in enumerate(paragraphs) if is_ingredient_sent(p)]
# ingredients = [paragraphs[idx] for idx in ingredients_indexes]

# print(f"Serves: {serving_amount}")

# print("Ingredients:")
# for ingredient in ingredients:
#     print(ingredient[0])

# unprocessed_paras = [
#     p
#     for idx, p in enumerate(paragraphs)
#     if idx not in ingredients_indexes and idx != serving_index
# ]

# for p in unprocessed_paras:
#     print(f"\nParagraph: {p}")
#     p_type = input("What type of paragraph is this? ")
