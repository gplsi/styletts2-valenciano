from transformers import BertTokenizer
import phonemizer

tokenizer = BertTokenizer.from_pretrained('javiimts/bert-ca-va-tokenizer')

# Fonemizamos el texto de ejemplo
text = 'Hola, com estàs? Aquest és un text de prova.'

global_phonemizer_valencia = phonemizer.backend.EspeakBackend(
    language="ca-va",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)   

phonemized_output = global_phonemizer_valencia.phonemize(
    [text],
    strip=True,
    njobs=1,
)

print("Texto original:")
print(text)
print("\nTexto fonemizado:")
print(phonemized_output[0])
print("Longitud fonemizada:", len(phonemized_output[0]))

encoded = tokenizer(
    phonemized_output[0])

print("\nTokens y IDs:")
print(len(encoded['input_ids']))