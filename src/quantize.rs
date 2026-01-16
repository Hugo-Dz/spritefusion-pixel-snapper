use crate::config::Config;
use crate::error::Result;
use image::RgbaImage;
use palette::Srgb;

use rayon::prelude::*;

/// Maximum palette size
pub const MAX_PALETTE: usize = 64;

/// Quantized image with pre-built palette and indexed buffer for efficient resampling
pub struct QuantizedImage {
    pub img: RgbaImage,
    pub palette: Vec<[u8; 4]>,
    pub indexed: Vec<u8>, // Each pixel maps to palette index (255 = transparent)
}

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

/// Quantizes the image and returns a QuantizedImage with pre-built palette and indexed buffer.
/// This eliminates the need for resample to rebuild the palette.
pub fn quantize_image(img: &RgbaImage, config: &Config) -> Result<QuantizedImage> {
    let k_target = if config.k_colors == 0 {
        auto_detect_k_colors(img)
    } else {
        config.k_colors
    };

    let width = img.width() as usize;
    let height = img.height() as usize;
    let in_samples = img.as_flat_samples().samples;
    let in_stride = width * 4;

    // Check for custom palette
    if let Some(palette_path) = &config.palette {
        let p_img = image::open(palette_path)
            .map_err(|e| {
                crate::error::PixelSnapperError::ProcessingError(format!(
                    "Failed to load palette: {}",
                    e
                ))
            })?
            .to_rgba8();

        let mut unique_colors = std::collections::HashSet::new();
        let mut palette_vec = Vec::new();

        for p in p_img.pixels() {
            if p[3] > 128 {
                // Ignore transparent pixels
                // Check if color already exists
                if unique_colors.insert(p.0) {
                    palette_vec.push(p.0);
                    if palette_vec.len() >= MAX_PALETTE {
                        break;
                    }
                }
            }
        }

        if palette_vec.is_empty() {
            return Err(crate::error::PixelSnapperError::ProcessingError(
                "Palette image contains no opaque pixels".to_string(),
            ));
        }

        // Map pixels to nearest palette color
        let mut indexed = vec![255u8; width * height];

        // Use Srgb for distance calculation
        let palette_srgb: Vec<Srgb> = palette_vec
            .iter()
            .map(|p| {
                Srgb::new(
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                )
            })
            .collect();

        // Parallel processing for nearest neighbor mapping
        indexed
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row_indices)| {
                for x in 0..width {
                    let base = y * in_stride + x * 4;
                    let a = in_samples[base + 3];
                    if a > 0 {
                        let r = in_samples[base] as f32 / 255.0;
                        let g = in_samples[base + 1] as f32 / 255.0;
                        let b = in_samples[base + 2] as f32 / 255.0;
                        let input_color = Srgb::new(r, g, b);

                        // Find nearest
                        let mut min_dist = f32::MAX;
                        let mut best_idx = 0;

                        for (i, p_color) in palette_srgb.iter().enumerate() {
                            // Simple Euclidean distance squared
                            let dr = input_color.red - p_color.red;
                            let dg = input_color.green - p_color.green;
                            let db = input_color.blue - p_color.blue;
                            let dist = dr * dr + dg * dg + db * db;
                            if dist < min_dist {
                                min_dist = dist;
                                best_idx = i;
                            }
                        }

                        row_indices[x] = best_idx as u8;
                    }
                }
            });

        // Write output pixels based on indexed
        let out_samples: Vec<u8> = indexed
            .par_iter()
            .map(|&idx| {
                if idx == 255 {
                    // Transparent
                    vec![0, 0, 0, 0]
                } else {
                    let p = palette_vec[idx as usize];
                    vec![p[0], p[1], p[2], 255]
                }
            })
            .flatten()
            .collect();

        return Ok(QuantizedImage {
            img: RgbaImage::from_raw(width as u32, height as u32, out_samples).unwrap(),
            palette: palette_vec,
            indexed,
        });
    }

    // Fast path: Build palette and indexed buffer directly for low-color images
    let mut color_to_idx = std::collections::HashMap::with_capacity(MAX_PALETTE);
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(MAX_PALETTE);
    let mut indexed = vec![255u8; width * height];
    let mut exceeded_k = false;

    for y in 0..height {
        for x in 0..width {
            let base = y * in_stride + x * 4;
            let a = in_samples[base + 3];
            if a > 0 {
                let pixel = [
                    in_samples[base],
                    in_samples[base + 1],
                    in_samples[base + 2],
                    a,
                ];
                let idx = if let Some(&existing) = color_to_idx.get(&pixel) {
                    existing
                } else {
                    if palette.len() >= MAX_PALETTE || palette.len() > k_target {
                        exceeded_k = true;
                        255
                    } else {
                        let new_idx = palette.len() as u8;
                        color_to_idx.insert(pixel, new_idx);
                        palette.push(pixel);
                        new_idx
                    }
                };
                indexed[y * width + x] = idx;
            }
        }
        if exceeded_k && palette.len() > k_target {
            break; // Need K-means, stop early
        }
    }

    // If unique colors ≤ k_target, we're done - no K-means needed
    if !exceeded_k || palette.len() <= k_target {
        return Ok(QuantizedImage {
            img: img.clone(),
            palette,
            indexed,
        });
    }

    // Need K-means clustering
    let opaque_indices: Vec<usize> = (0..width * height)
        .filter(|&i| in_samples[i * 4 + 3] > 0)
        .collect();

    if opaque_indices.is_empty() {
        return Ok(QuantizedImage {
            img: img.clone(),
            palette: vec![],
            indexed: vec![255u8; width * height],
        });
    }

    // Convert to Srgb floats in parallel (ignoring alpha for clustering)
    let pixels: Vec<Srgb> = opaque_indices
        .par_iter()
        .map(|&i| {
            let base = i * 4;
            Srgb::new(
                in_samples[base] as f32 / 255.0,
                in_samples[base + 1] as f32 / 255.0,
                in_samples[base + 2] as f32 / 255.0,
            )
        })
        .collect();

    let k = k_target.min(pixels.len());
    let result = kmeans_colors::get_kmeans_hamerly(
        k,
        config.max_kmeans_iterations,
        0.001, // Tighter convergence for RGB
        false,
        &pixels,
        config.k_seed,
    );

    // Build new palette from centroids
    let new_palette: Vec<[u8; 4]> = result
        .centroids
        .iter()
        .map(|&c| {
            [
                (c.red * 255.0).round() as u8,
                (c.green * 255.0).round() as u8,
                (c.blue * 255.0).round() as u8,
                255,
            ]
        })
        .collect();

    // Create output buffer and indexed buffer simultaneously
    let out_samples = in_samples.to_vec();
    let new_indexed = vec![255u8; width * height];

    opaque_indices
        .par_iter()
        .enumerate()
        .for_each(|(lab_idx, &pixel_idx)| {
            let centroid_idx = result.indices[lab_idx] as usize;
            let rgba = &new_palette[centroid_idx];
            let base = pixel_idx * 4;

            unsafe {
                let ptr = out_samples.as_ptr() as *mut u8;
                *ptr.add(base) = rgba[0];
                *ptr.add(base + 1) = rgba[1];
                *ptr.add(base + 2) = rgba[2];

                let idx_ptr = new_indexed.as_ptr() as *mut u8;
                *idx_ptr.add(pixel_idx) = centroid_idx as u8;
            }
        });

    let out_img =
        RgbaImage::from_raw(width as u32, height as u32, out_samples).ok_or_else(|| {
            crate::error::PixelSnapperError::ProcessingError(
                "Failed to create output image".to_string(),
            )
        })?;

    Ok(QuantizedImage {
        img: out_img,
        palette: new_palette,
        indexed: new_indexed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use image::Rgba;

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
        let img = RgbaImage::new(10, 10);
        let config = Config::default();
        let result = quantize_image(&img, &config).unwrap();
        assert_eq!(result.img.dimensions(), (10, 10));
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
        let p = result.img.get_pixel(0, 0);
        assert!(p[0] > 250);
        assert_eq!(result.palette.len(), 1);
    }
}
