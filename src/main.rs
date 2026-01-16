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

    let mut all_files = Vec::new();
    for path_str in &config.input_paths {
        let path = Path::new(path_str);
        if path.is_dir() {
            expand_directory(path, &mut all_files)?;
        } else {
            all_files.push(path.to_path_buf());
        }
    }

    if all_files.is_empty() {
        return Ok(());
    }

    let is_batch = all_files.len() > 1 || config.input_paths.iter().any(|p| Path::new(p).is_dir());

    for input_path in all_files {
        let input_path_str = input_path.to_string_lossy();
        println!("Processing: {}", input_path_str);

        let img_bytes = std::fs::read(&input_path).map_err(|e| {
            PixelSnapperError::ProcessingError(format!(
                "Failed to read input file {}: {}",
                input_path_str, e
            ))
        })?;

        let output_path_str = if is_batch {
            let out_dir = config.output.as_deref().unwrap_or(".");
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

fn expand_directory(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).map_err(|e| {
        PixelSnapperError::ProcessingError(format!(
            "Failed to read directory {}: {}",
            dir.display(),
            e
        ))
    })? {
        let entry = entry.map_err(|e| {
            PixelSnapperError::ProcessingError(format!("Failed to access entry: {}", e))
        })?;
        let path = entry.path();
        if path.is_dir() {
            expand_directory(&path, files)?;
        } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            let ext = ext.to_lowercase();
            match ext.as_str() {
                "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp" | "tiff" => {
                    files.push(path);
                }
                _ => {}
            }
        }
    }
    Ok(())
}
