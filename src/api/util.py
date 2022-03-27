import sys
import os
import pandas as pd
import subprocess
import json
import codecs
import unidecode

def process_text(text: str) -> str:
    text = text.replace('&quest', '?')
    
    return text


def normalize_text(txt: str) -> str:
    # TODO: REPLACE WITH YOUR OWN NORMALIZATION LOGIC HERE!!!!   
    valid_chars = (" ", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
    new_txt = unidecode.unidecode(txt.lower().strip())
    res_arr = []
    for c in new_txt:
        if c in valid_chars:
            res_arr.append(c)
        else:
            res_arr.append(' ')
    res = ''.join(res_arr).strip()    
    return ' '.join(res.split())