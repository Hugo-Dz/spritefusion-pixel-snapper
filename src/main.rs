use spritefusion_pixel_snapper::config::{parse_args, Config};
use spritefusion_pixel_snapper::error::PixelSnapperError;
use spritefusion_pixel_snapper::error::Result;
use spritefusion_pixel_snapper::process_image_bytes_common;

use std::path::Path;

fn main() -> Result<()> {
    let config = parse_args().unwrap_or_default();
    process_cli(&config)
}

fn process_cli(config: &Config) -> Result<()> {
    if config.input_paths.is_empty() {
        return Ok(());
    }

    let is_batch = config.input_paths.len() > 1;

    for input_path_str in &config.input_paths {
        let input_path = Path::new(input_path_str);
        println!("Processing: {}", input_path_str);

        let img_bytes = std::fs::read(input_path).map_err(|e| {
            PixelSnapperError::ProcessingError(format!(
                "Failed to read input file {}: {}",
                input_path_str, e
            ))
        })?;

        let output_path_str = if is_batch {
            let out_dir = config.output.as_deref().unwrap_or(".");
            // Ensure output directory exists
            if out_dir != "." {
                std::fs::create_dir_all(out_dir).map_err(|e| {
                    PixelSnapperError::ProcessingError(format!(
                        "Failed to create output directory {}: {}",
                        out_dir, e
                    ))
                })?;
            }
            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            format!("{}/{}_fixed.png", out_dir, stem)
        } else {
            config
                .output
                .clone()
                .unwrap_or_else(|| "output.png".to_string())
        };

        let output_bytes = process_image_bytes_common(&img_bytes, Some(config.clone()))?;

        std::fs::write(&output_path_str, output_bytes).map_err(|e| {
            PixelSnapperError::ProcessingError(format!(
                "Failed to write output file {}: {}",
                output_path_str, e
            ))
        })?;

        println!("Saved to: {}", output_path_str);
    }
    Ok(())
}
