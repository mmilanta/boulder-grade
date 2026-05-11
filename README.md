# Bouldering Grade Inference

A probabilistic model to infer "true" boulder difficulty from crowdsourced ascent
logs, accounting for the fact that we observe sends but not attempts.

## Motivation

Bouldering grades are notoriously noisy: they're set by first ascensionists,
revised by consensus, and vary by region, style, and era. Climbers logging
ascents on apps generate a rich dataset that implicitly encodes difficulty
information — strong climbers send hard boulders, and the grade at which
climbers stop sending reveals their ability.

We want to build a model that:

1. Infers each boulder's difficulty on a continuous latent scale, with
   calibrated uncertainty.
2. Infers each climber's ability on the same scale.
3. Identifies sandbagged or soft boulders (consensus grade vs. inferred grade).
4. Predicts send probability for any (climber, boulder) pair.

## Modeling Approach

We model each (climber, boulder) pair through a two-stage process:

P(logged ascent) = P(try) · P(send | try)


This decomposition addresses the central challenge: a missing entry in the ascent log is ambiguous between "didn't try" and "tried and failed." In our dataset we have:
* send (not flash): P(send) = P(try) · P(send | try) · (1 - P(flash | send))
* flash/onsight attempts: P(flash) = P(try) · P(send | try) · P(flash | send)
* tried but not managed: P(go) = P(try) · (1 - P(send | try))
* negative example (not in db): P(not-there) = (1 - P(try)) + P(try) · (1 - P(send | try))

### Model

For climber i and boulder j, with ability θᵢ and difficulty dⱼ:

- *Trying*: P(try) = σ(αᵢ + πⱼ - γ(θᵢ - dⱼ - μ)²)
  - αᵢ: climber prolificity (how much they log)
  - πⱼ: boulder popularity
  - The quadratic term encodes "climbers try things near their limit"
- *Sending given try*: P(send | try) = σ(θᵢ - dⱼ) (Rasch model)
- *Flash given send*: P(flash | send) = σ(θᵢ - dⱼ - β) (β > 0 is the flash penalty)

I would encode this as a pytorch model, where we have αᵢ and θᵢ for each climber, and dⱼ and πⱼ for each boulder. an then μ, γ, β as parameters to learn.

### The loss

I am not sure but I would use cross entropy loss, where the loss applied depends on weather the y is send or flash
