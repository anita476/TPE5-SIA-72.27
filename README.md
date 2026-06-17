# TPE5-SIA-72.27

Trabajo práctico 5 para Sistemas de Inteligencia Artificial: **Autoencoders**.

Implementa, sobre las imágenes binarias de caracteres de `data/font.h` (32 caracteres de 7×5 = 35 píxeles):

- un **Autoencoder básico** con espacio latente de 2 dimensiones,
- generación de **caracteres nuevos** a partir del espacio latente,
- un **Denoising Autoencoder** y el estudio de su robustez al ruido.

## Requisitos

```bash
pip install -r requirements.txt
```

## Estructura

```
configs/                      configs JSON de cada experimento
data/font.h                   dataset de caracteres (7x5)
src/
  autoencoders/
    Autoencoder.py            clase base (entrenamiento, Adam/SGD,
                              early stopping, restauración del mejor modelo)
    SimpleAutoencoder.py      autoencoder básico (forward/backprop)
    DenoisingAutoencoder.py   variante denoising (corrompe la entrada)
  utils/
    font_loader.py            parser de font.h
    config_loader.py          carga/validación de configs + grid
    grid_runner.py            grid search paralelo + plots + CSV
    single_run.py             una corrida de entrenamiento/evaluación
    noise.py                  ruido: gaussian / salt_pepper / masking
    denoising_eval.py         estudio de denoising por nivel de ruido
    latent_generate.py        generación desde el espacio latente
  test_autoencoder.py         entry point: entrenar/evaluar (con grid)
  generate_letter.py          entry point: generar letras nuevas
  denoising_experiment.py     entry point: entrenar DAE + estudio de ruido
```

## Uso

Todos los comandos se corren desde `src/`.

### 1a) Autoencoder básico (latente 2D, ≤1 píxel de error)

```bash
python test_autoencoder.py --config ../configs/default_simple.json
```

Aprende los 32 caracteres con error ≤ 1 píxel. Salidas en `output/simple/`:
`loss_1.png`, `latent_1.png` (espacio latente 2D), `errors_1.png`,
`reconstruction_1.txt`, `grid_results.csv`.

El config admite una sección `"grid"` (ver `configs/combination_simple.json`)
para barrer hiperparámetros en paralelo con `--workers N`.

### 1a-4) Generar una letra nueva desde el espacio latente

```bash
python generate_letter.py --config ../configs/default_simple.json --from c --to e
```

Salidas en `output/simple/`: `latent_grid.png` (grilla decodificada),
`latent_interpolation.png` (letras intermedias inexistentes en el dataset),
`latent_generated_point.png` (punto generado + glifo).

### 1b) Denoising Autoencoder

```bash
python denoising_experiment.py --config ../configs/default_denoising.json
```

Entrena corrompiendo la entrada (objetivo = imagen limpia) y luego evalúa la
reconstrucción a distintos niveles de ruido. Salidas en `output/denoising/`:
`loss.png`, `denoising_vs_noise.png`, `denoising_examples_<nivel>.png`,
`denoising_metrics.csv`.

## Opciones de config relevantes

- `layer_dims`: arquitectura; el valor central es el cuello de botella.
- `activation`, `optimizer` (`adam`/`sgd`), `lr`, `batch_size`, `epochs`, `seed`.
- `patience` / `min_delta`: early stopping (omitir `patience` lo desactiva).
- `threshold`, `max_errors`: binarización y tolerancia de píxeles.
- Denoising: `noise_type` (`gaussian`/`salt_pepper`/`masking`),
  `noise_level` (ruido de entrenamiento) y `noise_levels` (niveles a evaluar).

## Estado

- [x] grid support for config
- [x] parallel workers for seeds
- [x] csv output of epoch loss
- [x] early stopping for unchanging loss
- [x] best-model checkpoint (restore lowest-loss weights)
- [x] analyse latent space + generate new image from simple autoencoder
- [x] denoising
- [ ] maybe add softmax for threshold analysis
