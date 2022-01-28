from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.nlp.models.machine_translation import MTEncDecModel

translation_models = {
    'en': {
        'ru': 'nmt_en_ru_transformer6x6',
    },
    'ru': {
        'en': 'nmt_ru_en_transformer6x6',
    }
}

synthesis_models = {
    'en': {
        'spec_gen': 'tts_en_fastpitch',
        'vocoder': 'tts_hifigan'
    },
}

model_path = '/media/boris/F/'

translation_models['en']['ru'] = MTEncDecModel.from_pretrained(model_name="nmt_en_ru_transformer6x6")
translation_models['ru']['en'] = MTEncDecModel.from_pretrained(model_name="nmt_ru_en_transformer6x6")
synthesis_models['en']['spec_gen'] = FastPitchModel.from_pretrained('tts_en_fastpitch')
synthesis_models['en']['vocoder'] = HifiGanModel.from_pretrained('tts_hifigan')

# nmt_model_en_ru = MTEncDecModel.from_pretrained(model_name="nmt_en_ru_transformer6x6")
# nmt_model_ru_en = MTEncDecModel.from_pretrained(model_name="nmt_ru_en_transformer6x6")
# spec_gen = FastPitchModel.from_pretrained('tts_en_fastpitch')
# # spec_gen = FastPitchModel.load_from_checkpoint('tts_ru_fastpitch.ckpt').eval().cuda()
# vocoder = HifiGanModel.from_pretrained('tts_hifigan')