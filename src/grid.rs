use crate::config::Config;
use crate::error::{PixelSnapperError, Result};
use image::RgbaImage;
use std::cmp::Ordering;

use rayon::prelude::*;

#[allow(clippy::needless_range_loop)]
pub fn compute_profiles(img: &RgbaImage) -> Result<(Vec<f64>, Vec<f64>)> {
    let (w, h) = img.dimensions();
    let width = w as usize;
    let height = h as usize;

    if width < 3 || height < 3 {
        return Err(PixelSnapperError::InvalidInput(
            "Image too small (minimum 3x3)".to_string(),
        ));
    }

    // Use direct slice access to the image buffer for maximum performance
    let img_samples = img.as_flat_samples().samples;

    // Single-pass row-major scan using Rayon for maximum cache locality.
    // We aggregate horizontal and vertical projection sums simultaneously.
    let (col_proj, row_proj) = (1..height - 1)
        .into_par_iter()
        .fold(
            || (vec![0.0; width], vec![0.0; height]),
            |(mut cp, mut rp), y| {
                // Pre-calculate vertical offsets
                let row_curr_base = y * width * 4;
                let row_prev_base = (y - 1) * width * 4;
                let row_next_base = (y + 1) * width * 4;
                
                let mut row_diff_sum = 0.0;
                
                // For column projection: handle x in 1..width-1 within this row
                for x in 1..width - 1 {
                    let base = row_curr_base + x * 4;
                    
                    // Column contrib: |gray(x+1, y) - gray(x-1, y)|
                    // Fixed-point grayscale: (38*R + 75*G + 15*B) >> 7
                    let g_left = if img_samples[base - 4 + 3] == 0 { 0 } else {
                        (38 * img_samples[base - 4] as u32 + 
                         75 * img_samples[base - 3] as u32 + 
                         15 * img_samples[base - 2] as u32) >> 7
                    };
                    let g_right = if img_samples[base + 4 + 3] == 0 { 0 } else {
                        (38 * img_samples[base + 4] as u32 + 
                         75 * img_samples[base + 5] as u32 + 
                         15 * img_samples[base + 6] as u32) >> 7
                    };
                    cp[x] += (g_right as f64 - g_left as f64).abs();
                    
                    // Row contrib: |gray(x, y+1) - gray(x, y-1)|
                    let base_prev = row_prev_base + x * 4;
                    let base_next = row_next_base + x * 4;
                    
                    let g_top = if img_samples[base_prev + 3] == 0 { 0 } else {
                        (38 * img_samples[base_prev] as u32 + 
                         75 * img_samples[base_prev + 1] as u32 + 
                         15 * img_samples[base_prev + 2] as u32) >> 7
                    };
                    let g_bottom = if img_samples[base_next + 3] == 0 { 0 } else {
                        (38 * img_samples[base_next] as u32 + 
                         75 * img_samples[base_next + 1] as u32 + 
                         15 * img_samples[base_next + 2] as u32) >> 7
                    };
                    row_diff_sum += (g_bottom as f64 - g_top as f64).abs();
                }
                
                rp[y] = row_diff_sum;
                (cp, rp)
            },
        )
        .reduce(
            || (vec![0.0; width], vec![0.0; height]),
            |(mut cp1, mut rp1), (cp2, rp2)| {
                for (a, b) in cp1.iter_mut().zip(cp2.iter()) {
                    *a += b;
                }
                for (a, b) in rp1.iter_mut().zip(rp2.iter()) {
                    *a += b;
                }
                (cp1, rp1)
            },
        );

    Ok((col_proj, row_proj))
}

pub fn estimate_step_size(profile: &[f64], config: &Config) -> Option<f64> {
    if profile.is_empty() {
        return None;
    }

    let max_val = profile.iter().cloned().fold(0.0, f64::max);
    if max_val == 0.0 {
        return None; // Decide later
    }
    let threshold = max_val * config.peak_threshold_multiplier;

    let mut peaks = Vec::new();
    for i in 1..profile.len() - 1 {
        if profile[i] > threshold && profile[i] > profile[i - 1] && profile[i] > profile[i + 1] {
            peaks.push(i);
        }
    }

    if peaks.len() < 2 {
        return None;
    }

    let mut clean_peaks = vec![peaks[0]];
    for &p in peaks.iter().skip(1) {
        if p - clean_peaks.last().unwrap() >= config.peak_distance_filter {
            clean_peaks.push(p);
        }
    }

    if clean_peaks.len() < 2 {
        return None;
    }

    // Compute diffs
    let mut diffs: Vec<f64> = clean_peaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    // Median
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(diffs[diffs.len() / 2])
}

pub fn resolve_step_sizes(
    step_x_opt: Option<f64>,
    step_y_opt: Option<f64>,
    width: u32,
    height: u32,
    config: &Config,
) -> (f64, f64) {
    match (step_x_opt, step_y_opt) {
        (Some(sx), Some(sy)) => {
            let ratio = if sx > sy { sx / sy } else { sy / sx };
            if ratio > config.max_step_ratio {
                let smaller = sx.min(sy);
                (smaller, smaller)
            } else {
                let avg = (sx + sy) / 2.0;
                (avg, avg)
            }
        }

        (Some(sx), None) => (sx, sx),

        (None, Some(sy)) => (sy, sy),

        (None, None) => {
            let fallback_step =
                ((width.min(height) as f64) / config.fallback_target_segments as f64).max(1.0);
            (fallback_step, fallback_step)
        }
    }
}

pub fn stabilize_both_axes(
    profile_x: &[f64],
    profile_y: &[f64],
    raw_col_cuts: Vec<usize>,
    raw_row_cuts: Vec<usize>,
    width: usize,
    height: usize,
    config: &Config,
) -> (Vec<usize>, Vec<usize>) {
    let col_cuts_pass1 = stabilize_cuts(
        profile_x,
        raw_col_cuts.clone(),
        width,
        &raw_row_cuts,
        height,
        config,
    );
    let row_cuts_pass1 = stabilize_cuts(
        profile_y,
        raw_row_cuts.clone(),
        height,
        &raw_col_cuts,
        width,
        config,
    );

    // Check if the results are coherent
    let col_cells = col_cuts_pass1.len().saturating_sub(1).max(1);
    let row_cells = row_cuts_pass1.len().saturating_sub(1).max(1);
    let col_step = width as f64 / col_cells as f64;
    let row_step = height as f64 / row_cells as f64;

    let step_ratio = if col_step > row_step {
        col_step / row_step
    } else {
        row_step / col_step
    };

    if step_ratio > config.max_step_ratio {
        let target_step = col_step.min(row_step);

        let final_col_cuts = if col_step > target_step * 1.2 {
            snap_uniform_cuts(
                profile_x,
                width,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        } else {
            col_cuts_pass1
        };

        let final_row_cuts = if row_step > target_step * 1.2 {
            snap_uniform_cuts(
                profile_y,
                height,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        } else {
            row_cuts_pass1
        };

        (final_col_cuts, final_row_cuts)
    } else {
        (col_cuts_pass1, row_cuts_pass1)
    }
}

pub fn walk(profile: &[f64], step_size: f64, limit: usize, config: &Config) -> Result<Vec<usize>> {
    if profile.is_empty() {
        return Err(PixelSnapperError::ProcessingError(
            "Cannot walk on empty profile".to_string(),
        ));
    }

    let mut cuts = vec![0];
    let mut current_pos = 0.0;
    let search_window =
        (step_size * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val: f64 = profile.iter().sum::<f64>() / profile.len() as f64;

    while current_pos < limit as f64 {
        let target = current_pos + step_size;
        if target >= limit as f64 {
            cuts.push(limit);
            break;
        }

        let start_search = ((target - search_window) as usize).max((current_pos + 1.0) as usize);
        let end_search = ((target + search_window) as usize).min(limit);

        if end_search <= start_search {
            current_pos = target;
            continue;
        }

        let mut max_val = -1.0;
        let mut max_idx = start_search;
        for (i, &p) in profile
            .iter()
            .enumerate()
            .take(end_search)
            .skip(start_search)
        {
            if p > max_val {
                max_val = p;
                max_idx = i;
            }
        }

        if max_val > mean_val * config.walker_strength_threshold {
            cuts.push(max_idx);
            current_pos = max_idx as f64;
        } else {
            cuts.push(target as usize);
            current_pos = target;
        }
    }
    Ok(cuts)
}

/// Suggests a minimum number of grid segments based on image dimensions.
pub fn auto_detect_min_cuts(limit: usize) -> usize {
    // Heuristic: assume a minimum cell size of 4 for very small images,
    // up to a target of 16-32 segments for standard assets.
    (limit / 32).max(4).min(limit / 4).max(2)
}

pub fn stabilize_cuts(
    profile: &[f64],
    cuts: Vec<usize>,
    limit: usize,
    sibling_cuts: &[usize],
    sibling_limit: usize,
    config: &Config,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }

    let cuts = sanitize_cuts(cuts, limit);
    let auto_min = auto_detect_min_cuts(limit);
    let min_required = if config.min_cuts_per_axis == 0 {
        auto_min
    } else {
        config.min_cuts_per_axis
    }
    .max(2)
    .min(limit.saturating_add(1));
    let axis_cells = cuts.len().saturating_sub(1);
    let sibling_cells = sibling_cuts.len().saturating_sub(1);
    let sibling_has_grid =
        sibling_limit > 0 && sibling_cells >= min_required.saturating_sub(1) && sibling_cells > 0;
    let steps_skewed = sibling_has_grid && axis_cells > 0 && {
        let axis_step = limit as f64 / axis_cells as f64;
        let sibling_step = sibling_limit as f64 / sibling_cells as f64;
        let step_ratio = axis_step / sibling_step;
        step_ratio > config.max_step_ratio || step_ratio < 1.0 / config.max_step_ratio
    };
    let has_enough = cuts.len() >= min_required;

    if has_enough && !steps_skewed {
        return cuts;
    }

    let mut target_step = if sibling_has_grid {
        sibling_limit as f64 / sibling_cells as f64
    } else if config.fallback_target_segments > 1 {
        limit as f64 / config.fallback_target_segments as f64
    } else if axis_cells > 0 {
        limit as f64 / axis_cells as f64
    } else {
        limit as f64
    };
    if !target_step.is_finite() || target_step <= 0.0 {
        target_step = 1.0;
    }

    snap_uniform_cuts(profile, limit, target_step, config, min_required)
}

pub fn sanitize_cuts(mut cuts: Vec<usize>, limit: usize) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }

    let mut has_zero = false;
    let mut has_limit = false;

    for value in cuts.iter_mut() {
        if *value == 0 {
            has_zero = true;
        }
        if *value >= limit {
            *value = limit;
        }
        if *value == limit {
            has_limit = true;
        }
    }

    if !has_zero {
        cuts.push(0);
    }
    if !has_limit {
        cuts.push(limit);
    }

    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

pub fn snap_uniform_cuts(
    profile: &[f64],
    limit: usize,
    target_step: f64,
    config: &Config,
    min_required: usize,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    if limit == 1 {
        return vec![0, 1];
    }

    // Get desired cells
    let mut desired_cells = if target_step.is_finite() && target_step > 0.0 {
        (limit as f64 / target_step).round() as usize
    } else {
        0
    };
    desired_cells = desired_cells
        .max(min_required.saturating_sub(1))
        .max(1)
        .min(limit);

    let cell_width = limit as f64 / desired_cells as f64;
    let search_window =
        (cell_width * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };

    let mut cuts = Vec::with_capacity(desired_cells + 1);
    cuts.push(0);
    for idx in 1..desired_cells {
        let target = cell_width * idx as f64;
        let prev = *cuts.last().unwrap();
        if prev + 1 >= limit {
            break;
        }
        let mut start = ((target - search_window).floor() as isize)
            .max(prev as isize + 1)
            .max(0);
        let mut end = ((target + search_window).ceil() as isize).min(limit as isize - 1);
        if end < start {
            start = prev as isize + 1;
            end = start;
        }
        let start = start as usize;
        let end = end as usize;
        let mut best_idx = start.min(profile.len().saturating_sub(1));
        let mut best_val = -1.0;
        for (i, &v) in profile.iter().enumerate().take(end + 1).skip(start) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        let strength_threshold = mean_val * config.walker_strength_threshold;
        if best_val < strength_threshold {
            let mut fallback_idx = target.round() as isize;
            if fallback_idx <= prev as isize {
                fallback_idx = prev as isize + 1;
            }
            if fallback_idx >= limit as isize {
                fallback_idx = (limit as isize - 1).max(prev as isize + 1);
            }
            best_idx = fallback_idx as usize;
        }
        cuts.push(best_idx);
    }
    if *cuts.last().unwrap() != limit {
        cuts.push(limit);
    }
    cuts = sanitize_cuts(cuts, limit);
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_compute_profiles_small_image() {
        let img = RgbaImage::new(2, 2);
        let result = compute_profiles(&img);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_step_size_no_peaks() {
        let profile = vec![1.0, 1.0, 1.0, 1.0];
        let config = Config::default();
        let step = estimate_step_size(&profile, &config);
        assert!(step.is_none());
    }

    #[test]
    fn test_estimate_step_size_with_peaks() {
        // Peaks at 2, 6, 10 (diffs are 4)
        let mut profile = vec![0.0; 13];
        profile[2] = 10.0;
        profile[6] = 10.0;
        profile[10] = 10.0;
        let config = Config {
            peak_threshold_multiplier: 0.1,
            peak_distance_filter: 2,
            ..Config::default()
        };
        let step = estimate_step_size(&profile, &config);
        assert_eq!(step, Some(4.0));
    }

    #[test]
    fn test_sanitize_cuts() {
        let cuts = vec![5, 2, 8];
        let sanitized = sanitize_cuts(cuts, 10);
        assert_eq!(sanitized, vec![0, 2, 5, 8, 10]);
    }

    #[test]
    fn test_resolve_step_sizes_fallback() {
        let config = Config {
            fallback_target_segments: 10,
            ..Config::default()
        };
        let (sx, sy) = resolve_step_sizes(None, None, 100, 100, &config);
        assert_eq!(sx, 10.0);
        assert_eq!(sy, 10.0);
    }
}
