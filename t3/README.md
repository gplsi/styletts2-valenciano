# T03 — Selección y adaptación del transcriptor fonético

## Objetivo

Elegir y configurar una herramienta capaz de convertir texto ortográfico valenciano en una secuencia de fonemas IPA. Esta secuencia es la entrada de PL-BERT (T05) y de StyleTTS2 (T06).

## Decisión

Se utiliza la librería [`phonemizer`](https://github.com/bootphon/phonemizer) con el backend [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) configurada con `language="ca-va"` (catalán, variante valenciana).

```python
import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="ca-va",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)
```

Parámetros:

| Parámetro | Valor | Por qué |
|-----------|-------|---------|
| `language` | `ca-va` | Variante valenciana del catalán incluida en espeak-ng |
| `preserve_punctuation` | `True` | Mantiene comas, puntos, etc., relevantes para la prosodia |
| `with_stress` | `True` | Incluye marcas de acento primario (`ˈ`) y secundario (`ˌ`), claves para que StyleTTS2 aprenda la entonación |
| `language_switch` | `remove-flags` | Si aparece otro idioma en mitad del texto, elimina las marcas de cambio de idioma |

## Por qué se eligió esta opción

| Alternativa considerada | Resultado |
|-------------------------|-----------|
| **espeak-ng `ca-va`** (la elegida) | Soporta valenciano nativamente, marca acentos, gratuito, rápido, integrable con `phonemizer` |
| Modelos G2P entrenados ad-hoc | No disponibles para valenciano y entrenarlos requeriría un dataset alineado fonéticamente que no teníamos |
| espeak `ca` (catalán estándar) | No produce las realizaciones específicas del valenciano (vocales tónicas, neutralizaciones de vocales átonas, etc.) |

El proyecto Aina (BSC) usa el mismo backend en su [demo de transcripción fonética](https://huggingface.co/spaces/projecte-aina/transcripcio-fonetica-catala), lo que valida la elección.

## Importante: fonemización a nivel de frase

Hay que pasar las **frases completas** al fonemizador, no palabra por palabra. espeak-ng tiene en cuenta efectos fonéticos entre palabras. Si se fonemizan palabras aisladas, esos fenómenos se pierden y el modelo aprende un valenciano artificial.

Esto se respeta tanto en T02 (preparación de datos para StyleTTS2) como en T05 (preparación de datos para PL-BERT).

## Inputs

Texto ortográfico, por ejemplo:
```
Hola, com estàs? Aquest és un text de prova.
```

## Outputs

Secuencia de fonemas IPA con marcas de acento, por ejemplo:
```
ˈɔla, kˈom astˈas? akˈed ˈɛz un tˈeʒ de pɾˈɔva.
```

El alfabeto resultante (178 símbolos: IPA + puntuación + letras + padding) es el vocabulario fijo del `TextCleaner` de StyleTTS2 ([`StyleTTS2/meldataset.py`](../StyleTTS2/meldataset.py)).

## Cómo replicar / probar

### Instalación de espeak-ng

**Recomendación**: compilar **espeak-ng desde el código fuente** (versión `master` o `≥ 1.52`). Es lo que se hizo en este proyecto.

#### Verificar la instalación actual

```bash
espeak-ng --version          # Debe ser >= 1.51 (idealmente >= 1.52)
espeak-ng --voices=ca        # Debe listar 'ca-va Catalan_(Valencian)'
```

#### Compilar desde el código fuente (recomendado)

Clonar el repositorio oficial de espeak-ng y compilarlo:

```bash
# Dependencias de compilación (Ubuntu/Debian)
sudo apt-get install autoconf automake libtool pkg-config make git

# Clonar y compilar
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
./autogen.sh
./configure --prefix=/usr/local
make
sudo make install
sudo ldconfig

# Verificar
/usr/local/bin/espeak-ng --version
/usr/local/bin/espeak-ng --voices=ca | grep ca-va
```

> Se instala en `/usr/local/bin/espeak-ng`. Si en tu sistema ya hay `/usr/bin/espeak-ng` (de apt), añadir `/usr/local/bin` al PATH antes que `/usr/bin` o usar la ruta absoluta.

### Probar el fonemizador

```bash
# 1. Instalar la librería Python
pip install phonemizer

# 2. Ejecutar el script de prueba
python phonemizer_test.py
```

Salida esperada:
```
Fonetizador 'ca-va' (valencià) cargado.
Texto original:
Hola, com estàs? Aquest és un text de prova.

Texto fonemizado:
ˈɔla, kˈom astˈas? akˈed ˈɛz un tˈeʒ de pɾˈɔva.
```

## Archivos

- [`phonemizer_test.py`](phonemizer_test.py) — script mínimo que carga el fonemizador y prueba con una frase.

## Dependencias

- `espeak-ng` instalado en el sistema, con soporte de la variante objetivo (`ca-va` en nuestro caso).
- `phonemizer` (Python).

## Referencias

- [phonemizer](https://github.com/bootphon/phonemizer)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)
- [Demo Aina BSC](https://huggingface.co/spaces/projecte-aina/transcripcio-fonetica-catala)
