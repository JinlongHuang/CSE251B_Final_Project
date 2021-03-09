################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

# See this for input references - https://urldefense.com/v3/__https://www.nltk.org/api/nltk.translate.html*nltk.translate.bleu_score.sentence_bleu__;Iw!!Mih3wA!Wq83jaNrHwIpeQ6Nhqht_dgBzF3jc5LYS3MZ-AYh6xIYveu-JINbTzAkxsclLYU2$
# A Caption should be a list of strings.
# Reference Captions are list of actual captions - list(list(str))
# Predicted Caption is the string caption based on your model's output - list(str)
# Make sure to process your captions before evaluating bleu scores -
# Converting to lower case, Removing tokens like <start>, <end>, padding etc.

def bleu1(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)


def bleu4(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)


def clean_caption(indx, tokenizer):
    # Transforms an array of vocab indexes into an array of words with start, end, and pad tags removed. 
    clean_text = []

    for i in range(len(indx)):
        new_sentence = []

        for p in indx[i]: 
            word = tokenizer._convert_id_to_token(int(p))
            if not (('[' in word) or (']' in word) or ('##' in word)): 
                new_sentence.append(word)
            elif '[SEP]' in word: 
                break # End of sentence is reached x

        clean_text.append(new_sentence)

    return clean_text


def get_captions(img_ids, cocoTest):
    # Gets the captions from the coco object given image ids from the dataloader 

    all_caps = []
    for j in range(len(img_ids)):
        cap_for_1image = []
        one_image_info = cocoTest.imgToAnns[img_ids[j]] # A list of dictionary for one image, with keys 'image_id', id', 'caption'.
        all_caps.append([])
        for k in range(len(one_image_info)):
            cap_for_1image.append(one_image_info[k]['caption'].lower().split()) # (batch_size, 5, max_length)

        all_caps[-1] = cap_for_1image

    return all_caps


def calculate_bleu(bleu_func, clean_preds_text, clean_targets_text):
    # Calculates the aggregate bleu value in the bleu and clean targets text 
    b = 0

    for pred_text, targets_text in zip(clean_preds_text, clean_targets_text):
        # clean_pred_text: (batch_size, max_length)
        # clean_targets_text: (batch_size, 5, max_length)
        b += bleu_func(targets_text, pred_text)

    return b


def stochastic_generation(outputs_raw, temperature):
    # Calculate weighted softmax
    s = torch.nn.Softmax(dim=2)
    weighted_softmax = s(outputs_raw / temperature)

    # Sample from probability distribution
    prob_dist = torch.distributions.Categorical(weighted_softmax)
    preds = prob_dist.sample()

    return preds