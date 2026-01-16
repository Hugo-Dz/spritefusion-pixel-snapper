# Sprite Fusion Pixel Snapper

**Online version**: [spritefusion.com/pixel-snapper](https://spritefusion.com/pixel-snapper)

A multi-threaded grid alignment and color quantization tool for pixel art. It fixes naming, alignment, and palette inconsistencies in generated or scanned assets.

<img src="./static/hero.png" alt="Pixel Snapper" style="width: 100%; image-rendering: pixelated;">

## Features

- **Multi-threaded**: Parallelized grid detection and quantization using `rayon`.
- **Fast**: ~0.96s for 1.3M pixels on Apple Silicon (M-series).
- **SIMD Optimized**: Fixed-point grayscale and direct memory access for efficient execution.
- **Batch Mode**: Recursive directory processing for large asset sets.
- **Auto-detection**: Heuristic-based parameter selection for color count and grid size.
- **Robust**: Handles non-uniform grids and noise in generated assets.

## Why?

AI generation models often produce assets that are not perfectly aligned to a grid. Pixels may have varying sizes, and palettes can be noisy.

Pixel Snapper:
- Snaps pixels to a uniform, stabilized grid.
- Quantizes colors to a clean, perceptual palette (Lab space).
- Preserves detail like dithering while fixing structural drift.

<img src="./static/details.png" alt="Details" style="width: 100%; image-rendering: pixelated;">

## Usage

### CLI

Requires [Rust](https://www.rust-lang.org/).

```bash
git clone https://github.com/sabraman/spritefusion-pixel-snapper.git
cd spritefusion-pixel-snapper
cargo build --release
```

**Single Image:**
```bash
./target/release/spritefusion-pixel-snapper input.png --output output.png
```

**Recursively process a folder:**
```bash
./target/release/spritefusion-pixel-snapper ./assets --output ./processed
```

**Specify color count:**
```bash
./target/release/spritefusion-pixel-snapper input.png --output output.png --k-colors 16
```

### WASM

Build the WASM module for browser integration:

```bash
wasm-pack build --target web --release
```

## Architecture

The project is structured as a library (`src/lib.rs`) with a CLI implementation in `src/main.rs`.

- `src/grid.rs`: Single-pass cache-coherent grid detection.
- `src/quantize.rs`: Parallel K-Means clustering in Lab space (Hamerly algorithm).
- `src/resample.rs`: Linear-time pixel frequency resolution.

## Acknowledgments

Pixel Snapper is a [Sprite Fusion](https://spritefusion.com) project.

<img src="./static/spritefusion.webp" alt="Sprite Fusion" style="width: 100%;">

## License

MIT License [Hugo Duprez](https://www.hugoduprez.com/)
