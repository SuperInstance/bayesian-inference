# bayesian-inference

**Bayesian posterior updating for Beta-Binomial models and Gaussian Naive Bayes classification in Rust.**

Bayesian inference is the mathematical framework for updating beliefs in the presence of evidence. Given a prior distribution P(θ), observed data D, and a likelihood P(D|θ), Bayes' theorem computes the posterior:

> P(θ|D) = P(D|θ) · P(θ) / P(D)

This crate implements two foundational Bayesian methods: **Beta-Binomial conjugate updating** for binary outcome inference, and **Gaussian Naive Bayes** for classification.

## Why It Matters

Bayesian methods are essential when:

- **Sample sizes are small** — Frequentist estimates have wide confidence intervals; Bayesian priors regularize.
- **Sequential updating matters** — As new data arrives, the posterior becomes the next prior, enabling online learning.
- **Uncertainty quantification is required** — Bayesian methods produce full posterior distributions, not point estimates.
- **Decision-making under uncertainty** — Posterior probabilities directly inform optimal decisions (expected utility maximization).

Applications:

- **A/B testing** — Beta-Binomial models are the standard for conversion rate estimation with sequential stopping.
- **Spam filtering** — Naive Bayes was the first effective spam filter (Graham, 2002) and remains a strong baseline.
- **Medical diagnosis** — Posterior probability of disease given test results and prevalence (prior).
- **Reinforcement learning** — Thompson sampling uses posterior samples for exploration-exploitation.

## How It Works

### Beta-Binomial Model

The Beta distribution Beta(α, β) is the **conjugate prior** for the Binomial likelihood. This means the posterior is also Beta:

> Prior: θ ~ Beta(α₀, β₀)  
> Data: k successes in n trials  
> Posterior: θ | data ~ Beta(α₀ + k, β₀ + n - k)

Conjugacy is mathematically elegant: no integration needed, just parameter addition. The posterior mean and variance:

> E[θ] = α / (α + β)  
> Var[θ] = αβ / ((α + β)² (α + β + 1))

With a uniform prior Beta(1, 1) and 22 successes out of 35 trials:

> Posterior: Beta(23, 14)  
> Mean: 23/37 ≈ 0.622  
> Std: √(23·14 / (37²·38)) ≈ 0.078

This means: "There's a 62.2% success rate, ±7.8%."

### Log-Gamma Function

The Beta PDF normalization constant requires Gamma functions:

> B(α, β) = Γ(α)Γ(β) / Γ(α + β)

We use the **Lanczos approximation** for log-Γ:

> ln Γ(z) ≈ (z + 0.5) ln(z + 4.5) - (z + 4.5) + ln(√(2π) · (c₀ + Σ cᵢ/(z+i)))

with coefficients from Numerical Recipes. This achieves ~15 decimal digits of accuracy for z > 0.5.

### Gaussian Naive Bayes

For classification with continuous features, we model each feature as Gaussian per class:

> P(xⱼ | y=c) = (1 / (σⱼc √(2π))) · exp(-(xⱼ - μⱼc)² / (2σⱼc²))

The **naive** assumption: features are conditionally independent given the class:

> P(x₁, ..., x_d | y=c) = Πⱼ P(xⱼ | y=c)

Classification via maximum a posteriori (MAP):

> ŷ = argmax_c [ ln P(y=c) + Σⱼ ln P(xⱼ | y=c) ]

Working in log-space prevents floating-point underflow when multiplying many probabilities.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Posterior update (k, n) | O(1) | O(1) |
| Beta PDF evaluation | O(1) | O(1) |
| `lgamma(x)` | O(1) | O(1) — 6-term Lanczos |
| Naive Bayes predict (d features, C classes, N training) | O(N·d) for training; O(C·d) per prediction | O(C·d) |
| Gaussian PDF | O(1) | O(1) |

## Quick Start

```rust
// This crate runs as a binary demo
// Run: cargo run

// Beta-Binomial updating:
// Prior: Beta(1, 1) = Uniform
// After observing 3/5, 7/10, 12/20:
// Posterior: Beta(24, 15), mean ≈ 0.615

// Naive Bayes classification:
// 6 training samples in 2 classes, 2 features
// Test point (1.1, 2.0) → class 0
// Test point (4.2, 5.1) → class 1
```

### Using the Functions in Your Code

The `main.rs` contains standalone functions:

```rust
// Beta distribution helpers
fn beta_mean(alpha: f64, beta: f64) -> f64;
fn beta_var(alpha: f64, beta: f64) -> f64;
fn beta_pdf(x: f64, alpha: f64, beta: f64) -> f64;

// Naive Bayes
fn naive_bayes_predict(train: &[Sample], features: &[f64]) -> usize;

// Math utility
fn lgamma(x: f64) -> f64;  // Lanczos approximation
fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64;
```

## API

- **`lgamma(x: f64) → f64`** — Log-Gamma via 6-term Lanczos approximation
- **`beta_pdf(x, α, β) → f64`** — Beta distribution PDF
- **`beta_mean(α, β) → f64`** — α / (α + β)
- **`beta_var(α, β) → f64`** — αβ / ((α+β)²(α+β+1))
- **`gaussian_pdf(x, μ, σ) → f64`** — Normal distribution PDF
- **`naive_bayes_predict(train, features) → usize`** — Gaussian NB classification
- **`Sample`** — { features: Vec\<f64\>, label: usize }

## Architecture Notes

The γ+η=C identity: γ (generative capacity) is the model's ability to represent diverse distributions — Beta supports any shape on [0,1], Gaussian supports any real-valued feature. η (evaluative depth) is the inference quality — conjugate updating is exact (no approximation), Naive Bayes trades η for simplicity via the independence assumption. C = predictive accuracy, which is maximized when the model's γ matches the data's structure and η (inference) exploits it fully.

## References

1. Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. — Standard Bayesian statistics reference.
2. Press, W. et al. (2007). *Numerical Recipes* (3rd ed.), §6.1. — Lanczos approximation for Gamma function.
3. Bishop, C. (2006). *Pattern Recognition and Machine Learning*, §4.2. — Naive Bayes and Gaussian classifiers.
4. Graham, P. (2002). "A Plan for Spam." — Practical Naive Bayes for spam filtering.

## License

MIT
