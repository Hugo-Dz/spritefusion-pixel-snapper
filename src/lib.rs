pub mod config;
pub mod error;
pub mod grid;
pub mod quantize;
pub mod resample;

use image::GenericImageView;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub use crate::config::Config;
pub use crate::error::{PixelSnapperError, Result};

pub fn process_image_bytes_common(input_bytes: &[u8], config: Option<Config>) -> Result<Vec<u8>> {
    let config = config.unwrap_or_default();

    let img = image::load_from_memory(input_bytes)?;
    let (width, height) = img.dimensions();

    validate_image_dimensions(width, height)?;

    let rgba_img = img.to_rgba8();

    // Quantize and get pre-built palette + indexed buffer
    let quantized = crate::quantize::quantize_image(&rgba_img, &config)?;

    let (profile_x, profile_y) = crate::grid::compute_profiles(&quantized.img)?;

    let step_x_opt = crate::grid::estimate_step_size(&profile_x, &config);
    let step_y_opt = crate::grid::estimate_step_size(&profile_y, &config);

    let (step_x, step_y) =
        crate::grid::resolve_step_sizes(step_x_opt, step_y_opt, width, height, &config);

    let raw_col_cuts = crate::grid::walk(&profile_x, step_x, width as usize, &config)?;
    let raw_row_cuts = crate::grid::walk(&profile_y, step_y, height as usize, &config)?;

    let (col_cuts, row_cuts) = crate::grid::stabilize_both_axes(
        &profile_x,
        &profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width as usize,
        height as usize,
        &config,
    );

    // Use pre-built palette and indexed buffer - no redundant work!
    let output_img = crate::resample::resample(
        &quantized.img,
        &col_cuts,
        &row_cuts,
        &quantized.palette,
        &quantized.indexed,
    )?;

    let mut output_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut output_bytes);
    output_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(PixelSnapperError::ImageError)?;

    Ok(output_bytes)
}

/// WASM entry point
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image(
    input_bytes: &[u8],
    k_colors: Option<u32>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "k_colors must be greater than 0",
            ));
        }
        config.k_colors = k as usize;
    }

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

pub fn validate_image_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions cannot be zero".to_string(),
        ));
    }
    if width > 10000 || height > 10000 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions too large (max 10000x10000)".to_string(),
        ));
    }
    Ok(())
}
