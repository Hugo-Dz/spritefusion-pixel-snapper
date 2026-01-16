use spritefusion_pixel_snapper::config::{parse_args, Config};
use spritefusion_pixel_snapper::error::PixelSnapperError;
use spritefusion_pixel_snapper::error::Result;
use spritefusion_pixel_snapper::process_image_bytes_common;

fn main() -> Result<()> {
    let config = parse_args().unwrap_or_default();
    process_cli(&config)
}

fn process_cli(config: &Config) -> Result<()> {
    println!("Processing: {}", config.input_path);

    let img_bytes = std::fs::read(&config.input_path).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to read input file: {}", e))
    })?;

    let output_bytes = process_image_bytes_common(&img_bytes, Some(config.clone()))?;

    std::fs::write(&config.output_path, output_bytes).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to write output file: {}", e))
    })?;

    println!("Saved to: {}", config.output_path);
    Ok(())
}
