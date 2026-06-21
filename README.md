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
output/                       resultados (plots, CSV, reconstrucciones)
scripts/                      entry points ejecutables
  _bootstrap.py               agrega src/ al path y resuelve rutas a la raíz
  main.py                     demo rápida
  plot_dataset.py             grilla de los 32 caracteres de entrada
  test_autoencoder.py         entrenar/evaluar (multi-seed; grid opcional)
  generate_letter.py          generar letras nuevas (por seed)
  denoising_experiment.py     entrenar DAE + estudio de ruido (multi-seed)
  compare_experiment.py       comparar arquitecturas/optimizadores (multi-seed)
  compare_denoising.py        comparar tipos de ruido (multi-seed)
src/                          librería (paquetes importables)
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
    multiseed.py              corre N seeds y agrega media ± desv. + plots
    plot_style.py             estilo de gráficos compartido del TP
    noise.py                  ruido: gaussian / salt_pepper / masking
    denoising_eval.py         estudio de denoising por nivel de ruido
    latent_generate.py        generación desde el espacio latente
    comparison.py             estudio comparativo de variantes (multi-seed)
```

## Multi-seed por defecto

Todos los scripts de la consigna corren sobre **varias semillas** y reportan
**media ± desviación**, para que cualquier afirmación se base en una muestra y
no en una corrida afortunada. Las semillas se definen en el config como lista
explícita, p.ej. `"seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. La cantidad de
corridas **simultáneas** se elige con `--workers N` (default `1`, portable; no
está atada a la cantidad de CPUs de la máquina). Si el config no trae `"seeds"`,
se usa `--seeds N` como fallback generando `[1..N]`.

## Uso

Los comandos se corren desde la **raíz del proyecto**. Las rutas de `font` y
`out` de los configs se resuelven respecto de la raíz, así que las salidas
siempre van a `output/`.

### Dataset (grilla de los 32 caracteres)

```bash
python scripts/plot_dataset.py
```

Salida: `output/simple/dataset_grid.png`.

### 1a) Autoencoder básico (latente 2D, ≤1 píxel de error)

```bash
python scripts/test_autoencoder.py --config configs/default_simple.json --workers 8
```

Entrena el AE sobre las seeds del config y agrega media ± desv. Salidas en
`output/simple/`: `loss.png` (banda media±desv.), `errors.png` (error por
carácter con desvío), `latent_seeds.png` (espacio latente 2D por seed),
`summary.csv` (passed y error por seed + resumen).

El config también admite una sección `"grid"` (ver
`configs/combination_simple.json`): si tiene más de una combinación, se corre
el grid search clásico de hiperparámetros (un set de plots por corrida).

### 1a-4) Generar una letra nueva desde el espacio latente

```bash
python scripts/generate_letter.py --config configs/default_simple.json --from c --to e --workers 6
```

Salidas en `output/simple/`: `latent_interpolation_seeds.png` (interpolación
entre dos caracteres, una fila por seed; los pasos intermedios son letras
inexistentes en el dataset) y `latent_generated_point_seeds.png` (carácter
generado desde el centroide del latente, uno por seed).

### 1b) Denoising Autoencoder

```bash
python scripts/denoising_experiment.py --config configs/default_denoising.json --workers 8
```

Entrena corrompiendo la entrada (objetivo = imagen limpia) y evalúa la
reconstrucción a distintos niveles de ruido, agregando media ± desv. sobre las
seeds. Salidas en `output/denoising/`: `denoising_vs_noise.png` (banda),
`loss.png` y `denoising_examples_<nivel>.png` (seed representativa),
`denoising_metrics.csv`.

### Estudios comparativos

Comparar arquitecturas / optimizadores del autoencoder básico (mismo presupuesto):

```bash
python scripts/compare_experiment.py --config configs/compare_simple.json --workers 6
```

Define variantes en la clave `"variants"` del config; cada una se entrena sobre
todas las seeds. Salidas en `output/compare_simple/`: `compare_loss.png`
(bandas), `compare_metrics.png` (barras con barras de error), `compare_results.csv`
(media ± desv. por variante).

Comparar la robustez del denoising según el tipo de ruido
(gaussian / salt_pepper / masking):

```bash
python scripts/compare_denoising.py --config configs/default_denoising.json --workers 6
```

Salidas en `output/denoising/noise_compare/`:
`denoising_noise_comparison.png` (bandas por tipo), `denoising_noise_comparison.csv`.

## Opciones de config relevantes

- `layer_dims`: arquitectura; el valor central es el cuello de botella.
- `activation`, `optimizer` (`adam`/`sgd`), `lr`, `batch_size`, `epochs`, `seed`.
- `seeds`: lista de semillas para el estudio multi-seed, p.ej. `[1, 2, …, 10]`.
- `patience` / `min_delta`: early stopping (omitir `patience` lo desactiva).
- `threshold`, `max_errors`: binarización y tolerancia de píxeles.
- Denoising: `noise_type` (`gaussian`/`salt_pepper`/`masking`),
  `noise_level` (ruido de entrenamiento) y `noise_levels` (niveles a evaluar).

CLI común: `--workers N` (corridas simultáneas) y `--seeds N` (fallback si el
config no define `"seeds"`).

## Estado

- [x] grid support for config
- [x] parallel workers (selectable, config-driven seeds)
- [x] multi-seed aggregation (mean ± std) in every consigna script
- [x] csv output of epoch loss
- [x] early stopping for unchanging loss
- [x] best-model checkpoint (restore lowest-loss weights)
- [x] analyse latent space + generate new image from simple autoencoder
- [x] denoising + noise-type comparison
- [ ] maybe add softmax for threshold analysis
