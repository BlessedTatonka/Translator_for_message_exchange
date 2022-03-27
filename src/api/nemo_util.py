from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.nlp.models.machine_translation import MTEncDecModel

from scipy.io import wavfile

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
    'ru': {
        'spec_gen': 'tts_en_fastpitch',
        'vocoder': 'tts_hifigan'
    }
}

model_path = '/media/boris/F/'

is_initialized = False

initialize_on_startup = True

def models_init():
    global is_initialized
    if is_initialized:
        return True
    
    try:
        translation_models['en']['ru'] = MTEncDecModel.from_pretrained(model_name="nmt_en_ru_transformer6x6")
        translation_models['ru']['en'] = MTEncDecModel.from_pretrained(model_name="nmt_ru_en_transformer6x6")
        synthesis_models['en']['spec_gen'] = FastPitchModel.from_pretrained('tts_en_fastpitch').eval().cuda()
        synthesis_models['en']['vocoder'] = HifiGanModel.from_pretrained('tts_hifigan').eval()
        # synthesis_models['ru']['spec_gen'] = FastPitchModel.load_from_checkpoint(model_path + 'tts_ru_fastpitch.ckpt').eval().cuda()
        ckpt = "/media/boris/F/NeMo_own_research/tts/fastpitch_exp_manager/FastPitch/2022-01-30_04-40-08/checkpoints/FastPitch--v_loss=0.1730-epoch=13.ckpt"
        synthesis_models['ru']['spec_gen'] = FastPitchModel.load_from_checkpoint(ckpt).eval().cuda()
        synthesis_models['ru']['vocoder'] = synthesis_models['en']['vocoder']
        is_initialized = True
        return 200
    except:
        return 404

if initialize_on_startup:
    models_init()