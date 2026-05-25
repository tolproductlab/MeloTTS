try:
    from . import french
except ImportError:
    french = None
try:
    from . import english
except ImportError:
    english = None
try:
    from . import spanish
except ImportError:
    spanish = None
try:
    from . import chinese
except ImportError:
    chinese = None
try:
    from . import japanese
except ImportError:
    japanese = None
try:
    from . import chinese_mix
except ImportError:
    chinese_mix = None
try:
    from . import korean
except ImportError:
    korean = None

from . import cleaned_text_to_sequence
import copy

language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish}


def clean_text(text, language):
    language_module = language_module_map[language]
    if language_module is None:
        raise ImportError(f"Language {language} is not supported because its dependencies are missing.")
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = language_module_map[language]
    if language_module is None:
        raise ImportError(f"Language {language} is not supported because its dependencies are missing.")
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass