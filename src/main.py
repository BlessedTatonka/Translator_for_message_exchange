from typing import List, Optional
from fastapi import FastAPI, Query
import os
import numpy

import util
import nemo_util

app = FastAPI()

tmp_files_path = 'tmp_files/'

@app.get("/")
async def root():
    return {"message": "Translator for message exchange API is currently on."}

@app.get("/translate/{text}")
async def read_item(
    text: str = Query(
        None,
        title="Text to translate",
#         description="Language of the text",
    ),
    src_lang: str = Query(
        None,
        title="Source language",
        description="Language of the text",
    ),
    trg_lang: str = Query(
        None,
        title="Target language",
        description="Language of the translation",
    )):
    query = {
        "text": util.process_text(text),
        "src_lang": src_lang,
        "trg_lang": trg_lang
    }
    
    nmt_model = nemo_util.translation_models[query['src_lang']][query['trg_lang']]
    
    translation = nmt_model.translate(
        [query['text']],
         source_lang=query['src_lang'],
         target_lang=query['trg_lang'])
    
    return {"text": text, "translation": translation}

@app.get("/synthesize/{text}")
async def read_item(
    text: str,
    src_lang: str = Query(
        None,
        title="Source language",
        description="Language of the text",
    )):
    query = {
        "text": util.process_text(text),
        "src_lang": src_lang
    }
    
    spec_gen = nemo_util.synthesis_models[query['src_lang']]['spec_gen']
    vocoder = nemo_util.synthesis_models[query['src_lang']]['vocoder']
        
    parsed = spec_gen.parse(query['text'])
    spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
    waveform = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    
    audio = waveform[0].cpu().detach().numpy().tolist()
         
    return {"text": text, "audio": audio}