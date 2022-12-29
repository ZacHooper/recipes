# Use this file to do various processing tasks

# import json

# # merge topic training datasets
# with open("sample_data/topic_training.json", "r") as f:
#     topic_training = json.load(f)


# with open("sample_data/topic_training_2.json", "r") as f:
#     topic_training_2 = json.load(f)

# topic_training.extend(topic_training_2)

# with open("sample_data/topic_training.json", "w") as f:
#     f.write(json.dumps(topic_training))

from PIL import Image
from pytesseract import pytesseract
import spacy
import cv2
import numpy as np
from spacy.language import Language
from utils.text_processing import process_text, is_ingredient_sent
import json

# set to True if you want to manually classify text
USER_INPUT = True

### --------------------- ###
#       Load Models         #
### --------------------- ###
topic_model = spacy.load("textcat_output/model-best")
ner_model = spacy.load("ner_output/model-best")

### --------------------- ###
#       Image Import        #
### --------------------- ###

# Defining paths to tesseract.exe
# and the image we would be using
path_to_tesseract = r"/opt/homebrew/opt/tesseract/bin/tesseract"

images = [
    "sample_data/recipes/AJ-OPPP-greens_and_caramelised_tofu.JPG",
    "sample_data/recipes/AJ-OPPP-piquant_smoked_paprika_pasta_bake.JPG",
    "sample_data/recipes/AJ-OPPP-roasted_squash_with_lemongrass.JPG",
    "sample_data/recipes/AJ-OPPP-sweet_potato_and_miso_noodle_broth.JPG",
]

# image_path = r"sample_data/recipes/AJ_OPPP-winter_red_cabbage.JPG"

for image_path in images:

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
        elif len(paragraph) == 1:
            text = paragraph[0].text
        else:
            continue

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
            og_key = [k for k, v in potential_cats.items() if v == max_value][0]
            print(f"Original topic: {og_key}, score: {max_value}")
            max_key = input(
                f"Could not classify: {text} -- \nPlease enter a category: ('title', 'blurb', 'ingredient', 'method', 'other')\nInput topic: "
            )
        else:
            # get key of max value
            max_key = [k for k, v in potential_cats.items() if v == max_value][0]
        cats.append(
            {
                "text": text,
                "chosen_topic": max_key,
                "topic_score": max_value,
                "scores": potential_cats,
            }
        )

    print(
        json.dumps(
            [
                f"{cat['text'][:25]} - {cat['chosen_topic']} - {cat['topic_score']}"
                for cat in cats
            ],
            indent=4,
        )
    )

    with open("sample_data/topic_training_3.json", "r") as f:
        topic_training = json.load(f)

    # Spacy does not currently support partial scores of topics so we will hardcode the chosen topic to 1.0
    # and hardcode the other topics to 0.0. I will be manually updating the topic and scores if incorrect
    for cat in cats:
        cat["topic"] = {
            "blurb": 0.0,
            "ingredient": 0.0,
            "method": 0.0,
            "other": 0.0,
            "title": 0.0,
        }
        cat["topic"][cat["chosen_topic"]] = 1.0
        cat.pop("scores", None)
        cat.pop("chosen_topic", None)
        cat.pop("topic_score", None)

    topic_training.extend(cats)

    with open("sample_data/topic_training_3.json", "w") as f:
        f.write(json.dumps(topic_training))
