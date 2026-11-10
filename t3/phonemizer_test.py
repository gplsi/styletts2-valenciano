import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="ca-va", 
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)
print("Fonetizador 'ca-va' (valencià) cargado.")

text = "Hola, com estàs? Aquest és un text de prova."

phonemized_output = global_phonemizer.phonemize(
    [text],
    strip=True,
    njobs=1,
)

print("Texto original:")
print(text)
print("\nTexto fonemizado:")
print(phonemized_output[0])
print("Longitud fonemizada:", len(phonemized_output[0]))