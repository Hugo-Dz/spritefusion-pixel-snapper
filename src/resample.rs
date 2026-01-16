use crate::error::{PixelSnapperError, Result};
use image::{ImageBuffer, RgbaImage};
use rayon::prelude::*;
use std::collections::HashMap;

pub fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize]) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;

    // Parallelize processing of rows
    let rows_vec: Vec<&[usize]> = rows.windows(2).collect();
    let pixels: Vec<[u8; 4]> = rows_vec
        .into_par_iter()
        .flat_map(|w_y| {
            let ys = w_y[0];
            let ye = w_y[1];

            // Processing each column in the row
            cols.windows(2)
                .map(|w_x| {
                    let xs = w_x[0];
                    let xe = w_x[1];

                    if xe <= xs || ye <= ys {
                        return [0, 0, 0, 0];
                    }

                    let mut counts: HashMap<[u8; 4], usize> = HashMap::new();

                    for y in ys..ye {
                        for x in xs..xe {
                            if x < img.width() as usize && y < img.height() as usize {
                                let p = img.get_pixel(x as u32, y as u32).0;
                                *counts.entry(p).or_insert(0) += 1;
                            }
                        }
                    }

                    let mut best_pixel = [0, 0, 0, 0];
                    let mut candidates: Vec<([u8; 4], usize)> = counts.into_iter().collect();
                    candidates.sort_by(|a, b| {
                        b.1.cmp(&a.1).then_with(|| {
                            let sum_a: u32 = a.0.iter().map(|&v| v as u32).sum();
                            let sum_b: u32 = b.0.iter().map(|&v| v as u32).sum();
                            sum_b.cmp(&sum_a)
                        })
                    });

                    if let Some((p, _)) = candidates.first() {
                        best_pixel = *p;
                    }

                    best_pixel
                })
                .collect::<Vec<[u8; 4]>>()
        })
        .collect();

    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);
    for (i, &p) in pixels.iter().enumerate() {
        let x = i as u32 % out_w;
        let y = i as u32 / out_w;
        final_img.put_pixel(x, y, image::Rgba(p));
    }

    Ok(final_img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_insufficient_grid() {
        let img = RgbaImage::new(10, 10);
        let result = resample(&img, &vec![0], &vec![0, 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_simple_grid() {
        let mut img = RgbaImage::new(10, 10);
        // Fill the entire image with red
        for p in img.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let result = resample(&img, &vec![0, 10], &vec![0, 10]).unwrap();
        assert_eq!(result.dimensions(), (1, 1));
        assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
    }
}
