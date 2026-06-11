/// Bayesian Inference — posterior updating for a Beta-Binomial model.
///
/// Prior:  θ ~ Beta(α₀, β₀)
/// Data:   k successes in n trials
/// Posterior: θ | data ~ Beta(α₀ + k, β₀ + n − k)
///
/// Also includes naive Bayes classification on a small toy dataset.

use std::f64::consts::E;

// ── Beta distribution helpers ──

/// Log-Gamma via Stirling-style approximation (good enough for demo)
fn lgamma(x: f64) -> f64 {
    let mut s = 0.5 * (2.5066282746310005 / x);
    let mut t = x;
    for c in [76.18009172947146, -86.50532032941677, 24.01409824083091,
              -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5] {
        t += 1.0;
        s += c / t;
    }
    (x + 4.5) * (x + 0.5).ln() - (x + 4.5) + s.ln()
}

fn beta_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 { return 0.0; }
    let ln_norm = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta);
    (ln_norm + (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln()).exp()
}

fn beta_mean(alpha: f64, beta: f64) -> f64 {
    alpha / (alpha + beta)
}

fn beta_var(alpha: f64, beta: f64) -> f64 {
    let s = alpha + beta;
    alpha * beta / (s * s * (s + 1.0))
}

// ── Naive Bayes classifier ──

#[derive(Debug)]
struct Sample {
    features: Vec<f64>,
    label: usize,
}

fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

fn naive_bayes_predict(train: &[Sample], features: &[f64]) -> usize {
    let n_classes = train.iter().map(|s| s.label).max().unwrap() + 1;
    let n_feat = features.len();

    let mut best_class = 0;
    let mut best_score = f64::NEG_INFINITY;

    for c in 0..n_classes {
        let class_samples: Vec<&Sample> = train.iter().filter(|s| s.label == c).collect();
        let prior = class_samples.len() as f64 / train.len() as f64;
        let mut log_post = prior.ln();

        for j in 0..n_feat {
            let vals: Vec<f64> = class_samples.iter().map(|s| s.features[j]).collect();
            let mu = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / vals.len() as f64;
            let sigma = var.sqrt().max(0.01);
            log_post += gaussian_pdf(features[j], mu, sigma).ln();
        }

        if log_post > best_score {
            best_score = log_post;
            best_class = c;
        }
    }
    best_class
}

fn main() {
    // ── Part 1: Beta-Binomial posterior updating ──
    println!("═══ Beta-Binomial Bayesian Updating ═══\n");

    let mut alpha = 1.0; // Beta(1,1) = Uniform prior
    let mut beta_param = 1.0;

    let observations: &[(u32, u32)] = &[
        (3, 5),   // 3 successes out of 5
        (7, 10),  // 7 out of 10
        (12, 20), // 12 out of 20
    ];

    println!("Prior: Beta({alpha}, {beta_param}), mean = {:.4}", beta_mean(alpha, beta_param));
    println!();

    for (i, &(k, n)) in observations.iter().enumerate() {
        alpha += k as f64;
        beta_param += (n - k) as f64;
        let mean = beta_mean(alpha, beta_param);
        let var = beta_var(alpha, beta_param);
        println!("After obs {}: {} succ / {} trials", i + 1, k, n);
        println!("  Posterior: Beta({alpha:.0}, {beta_param:.0}), mean = {mean:.4}, std = {:.4}", var.sqrt());
    }

    // ── Part 2: Naive Bayes Classification ──
    println!("\n═══ Naive Bayes Classification ═══\n");

    let train = vec![
        Sample { features: vec![1.0, 2.0], label: 0 },
        Sample { features: vec![1.5, 1.8], label: 0 },
        Sample { features: vec![1.2, 2.1], label: 0 },
        Sample { features: vec![4.0, 5.0], label: 1 },
        Sample { features: vec![4.5, 4.8], label: 1 },
        Sample { features: vec![3.8, 5.2], label: 1 },
    ];

    let test = vec![
        vec![1.1, 2.0],
        vec![4.2, 5.1],
        vec![2.5, 3.5],
    ];

    for t in &test {
        let pred = naive_bayes_predict(&train, t);
        println!("Features ({}, {}) → class {pred}", t[0], t[1]);
    }
}
