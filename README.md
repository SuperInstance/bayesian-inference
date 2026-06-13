# Bayesian Inference

**A Rust library for Bayesian statistical inference** — implements Beta-Binomial posterior updating and Gaussian Naive Bayes classification using closed-form conjugate priors.

## Why It Matters

Bayesian inference is the mathematical framework for updating beliefs given evidence. Unlike frequentist statistics (which gives you p-values and confidence intervals), Bayesian methods give you **probability distributions over hypotheses** — directly answering the questions humans actually ask ("What's the probability the new feature improves conversion?").

The **Beta-Binomial model** is the "Hello World" of Bayesian inference. It models a binary outcome (coin flip, click/no-click, success/failure):

- The **prior** is a Beta(α, β) distribution encoding what you believe before seeing data
- The **likelihood** is Binomial: k successes in n trials
- The **posterior** is Beta(α + k, β + n − k) — conjugate priors make this a simple addition

The **posterior mean** `α / (α + β)` gives your best estimate of the success rate, and the **posterior variance** tells you how confident you are. As you collect more data, the variance shrinks — you become more certain.

**Naive Bayes classification** applies Bayes' theorem assuming feature independence. Despite the "naive" assumption, it works remarkably well for text classification, spam filtering, and medical diagnosis.

## How It Works

**Beta distribution math**: The Beta(α, β) PDF is `x^(α-1) · (1-x)^(β-1) / B(α, β)` where B is the Beta function. The library computes the log-Gamma function using a Lanczos approximation (6 terms, accurate to ~7 decimal digits) for the normalizing constant.

**Posterior updating**: Simply increment α by the number of successes and β by the number of failures. The mean and variance have closed forms: `mean = α/(α+β)`, `var = αβ / ((α+β)²(α+β+1))`.

**Naive Bayes**: For each class, compute the prior `P(class)` from training data, then for each feature compute `P(feature | class)` using a Gaussian (normal) distribution with class-conditional mean and variance. Multiply (add in log space) to get the posterior. The class with highest log-posterior wins.

## Quick Start

```rust
// Beta-Binomial updating (implemented in the binary)
// Prior: Beta(1, 1) = uniform
// After 3/5 successes: Beta(4, 3), mean = 4/7 ≈ 0.571
// After 7/10 more: Beta(11, 6), mean ≈ 0.647
// After 12/20 more: Beta(23, 14), mean ≈ 0.622

// Naive Bayes classification (from the demo):
// Training data:
//   Class 0: (1.0, 2.0), (1.5, 1.8), (1.2, 2.1)
//   Class 1: (4.0, 5.0), (4.5, 4.8), (3.8, 5.2)
// Test point (1.1, 2.0) → class 0
// Test point (4.2, 5.1) → class 1
```

## API

- **`lgamma(x)`** — Log-Gamma via Lanczos approximation
- **`beta_pdf(x, alpha, beta)`** — Beta distribution probability density
- **`beta_mean(alpha, beta)`** — Closed-form posterior mean
- **`beta_var(alpha, beta)`** — Closed-form posterior variance
- **`gaussian_pdf(x, mu, sigma)`** — Normal distribution PDF
- **`naive_bayes_predict(train, features)`** — Gaussian NB classification

## Architecture Notes

Provides the statistical inference primitives for SuperInstance analytics and experiment-evaluation pipelines. The Beta-Binomial model is used for A/B testing, conversion rate estimation, and Thompson sampling. See the [architecture overview](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## License

MIT
