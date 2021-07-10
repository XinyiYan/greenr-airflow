from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import emoji

tags_to_be_removed = {
        "<hashtag>", "</hashtag>", "<elongated>", "<repeated>", '<emphasis>',
        '<url>', '<email>', '<percent>', '<money>', '<phone>', '<user>',
        '<time>', '<date>', '<number>'
        }


def get_text_processor():
    return TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "elongated", "repeated", 'emphasis'}, # "allcaps",  , 'censored'

        fix_bad_unicode=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words
        spell_correction=True, #choose if you want to perform spell correction to the text

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )


def emoticons2text(sent):
    """
        Translate emoticions to text
    """
    return ' '.join([emoticons[word] if word in emoticons else word for word in sent.split()])


def remove_hashtag(sent):
    """
        Remove #hashtags 
    """
    return re.sub(r'#[A-Za-z0-9_]+', ' ', sent)


def post_process_tokens(tokens):
    """
        Remove <special tokens> introduced by TextPreProcessor
    """
    return [token for token in tokens if token not in tags_to_be_removed]


def remove_emoji(text):
   """
   remove all of emojis from text
   """
   text= emoji.demojize(text)
   text= re.sub(r'(:[!_\-\w]+:)', '', text)
   return ' '.join(text.split())


def extract_and_dedup_emoji(text):
    """
    Find emojis used in the given text, removing duplications
    """
    emojis = set(re.findall(r'(:[!_\-\w]+:)', emoji.demojize(text)))
    emojis = [i[1:-1] for i in emojis]
    return ' '.join(emojis)


def preprocess_text(text_processor, sent, keep_hashtags=True):
    """
    Text processing function:
        1. remove html, url, email, etc....
        2. translate emoticions to text
        3. translate emojis to text
    """
    sent = sent.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    sent = BeautifulSoup(sent, 'lxml').get_text()
    sent = emoticons2text(sent)
    if not keep_hashtags:
        sent  = remove_hashtag(sent)
    sent = " ".join(post_process_tokens(text_processor.pre_process_doc(sent)))
    sent = remove_emoji(sent)  + " " + extract_and_dedup_emoji(sent)
    return sent


