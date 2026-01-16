use crate::config::Config;
use crate::error::Result;
use image::RgbaImage;
use palette::{white_point::D65, FromColor, Lab, Srgba};

use rayon::prelude::*;

/// Detects the number of unique colors in the image to suggest an optimal K for quantization.
pub fn auto_detect_k_colors(img: &RgbaImage) -> usize {
    let mut unique_colors = std::collections::HashSet::new();
    for p in img.pixels() {
        if p[3] > 0 {
            unique_colors.insert(p.0);
            if unique_colors.len() >= 32 {
                return 32;
            }
        }
    }
    unique_colors.len().clamp(1, 32)
}

/// Quantizes the image to a fixed number of colors using K-means clustering.
/// Uses Lab color space for better perceptual color selection.
pub fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    let k_target = if config.k_colors == 0 {
        auto_detect_k_colors(img)
    } else {
        config.k_colors
    };


    let pixels: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();

    let opaque_indices: Vec<usize> = pixels
        .iter()
        .enumerate()
        .filter_map(|(i, p)| if p[3] > 0 { Some(i) } else { None })
        .collect();

    if opaque_indices.is_empty() {
        return Ok(img.clone());
    }

    // Convert opaque pixels to Lab for better perceptual clustering in parallel
    let lab_pixels: Vec<Lab<D65, f32>> = opaque_indices
        .par_iter()
        .map(|&i| {
            let p = pixels[i];
            let srgba = Srgba::new(
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                p[3] as f32 / 255.0,
            );
            Lab::from_color(srgba.into_linear::<f32, f32>())
        })
        .collect();

    let k = k_target.min(lab_pixels.len());
    let max_iter = config.max_kmeans_iterations;
    let converge = 0.01;
    let verbose = false;
    let seed = config.k_seed;

    // Perform K-means clustering in Lab space using the faster Hamerly algorithm
    let result = kmeans_colors::get_kmeans_hamerly(k, max_iter, converge, verbose, &lab_pixels, seed);

    // Prepare final image buffer
    let mut new_img = img.clone();
    let final_samples = new_img.as_flat_samples_mut().samples;

    // Map quantized pixels back in parallel results
    let results: Vec<(usize, [u8; 4])> = opaque_indices
        .par_iter()
        .enumerate()
        .map(|(idx, &pixel_idx)| {
            let centroid_idx = result.indices[idx];
            let lab_centroid = result.centroids[centroid_idx as usize];
            let srgba_centroid: Srgba = Srgba::from_color(lab_centroid);

            // Preserve original alpha
            let original_alpha = pixels[pixel_idx][3];
            
            let rgba = [
                (srgba_centroid.red * 255.0).round() as u8,
                (srgba_centroid.green * 255.0).round() as u8,
                (srgba_centroid.blue * 255.0).round() as u8,
                original_alpha,
            ];
            (pixel_idx, rgba)
        })
        .collect();

    // Fast linear update (sequential but memory-efficient)
    for (idx, rgba) in results {
        let base = idx * 4;
        final_samples[base] = rgba[0];
        final_samples[base + 1] = rgba[1];
        final_samples[base + 2] = rgba[2];
        final_samples[base + 3] = rgba[3];
    }

    Ok(new_img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_quantize_image_auto_colors() {
        let img = RgbaImage::new(10, 10);
        let config = Config {
            k_colors: 0,
            ..Config::default()
        };
        let result = quantize_image(&img, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantize_image_empty_image() {
        let img = RgbaImage::new(10, 10); // Fully transparent
        let config = Config::default();
        let result = quantize_image(&img, &config).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_quantize_image_single_color() {
        let mut img = RgbaImage::new(10, 10);
        for p in img.pixels_mut() {
            *p = Rgba([255, 0, 0, 255]);
        }
        let config = Config {
            k_colors: 1,
            ..Config::default()
        };
        let result = quantize_image(&img, &config).unwrap();
        // Lab conversion/quantization should be very close to original red
        let p = result.get_pixel(0, 0);
        assert!(p[0] > 250);
        assert!(p[1] < 10);
        assert!(p[2] < 10);
        assert_eq!(p[3], 255);
    }
}
