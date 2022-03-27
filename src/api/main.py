from fastapi import FastAPI, Query
from fastapi.responses import Response, FileResponse

import util
from nemo_util import *

import soundfile as sf
import noisereduce as nr


app = FastAPI()

tmp_files_path = 'tmp_files/'

@app.get("/")
async def root():
    return {"message": "Translator for message exchange API is currently on."}


@app.get("/init")
async def init():
    init_result = models_init()
    
    return {"status": "init_result"}


@app.get("/translate/{text}")
async def translate(
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
    
    nmt_model = translation_models[query['src_lang']][query['trg_lang']]
    
    translation = nmt_model.translate(
        [query['text']],
         source_lang=query['src_lang'],
         target_lang=query['trg_lang'])
    
    return {"text": text, "translation": translation}

@app.get("/synthesize/{text}")
async def synthesize(
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
    
    normalized_text = util.normalize_text(query['text'])
    
    try:
        with open(f'/audio/{normalized_text}.wav') as f:
            return FileResponse(f'audio/{normalized_text}.wav')
    except IOError:
        spec_gen = synthesis_models[query['src_lang']]['spec_gen']
        vocoder = synthesis_models[query['src_lang']]['vocoder']

        parsed = spec_gen.parse(normalized_text)
        spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
        waveform = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        audio = waveform[0].cpu().detach().numpy().tolist()

        audio = nr.reduce_noise(y=audio, sr=22050)
        audio = nr.reduce_noise(y=audio, sr=22050)
        
        sf.write(f'audio/{normalized_text}.wav', audio, 22050)

        return FileResponse(f'audio/{normalized_text}.wav')

@app.get('/dummy_audio')
async def dummy_audio():
    
#     samplerate, data = wavfile.read('dummy_audio.wav')
    
    return FileResponse('dummy_audio.wav')

    