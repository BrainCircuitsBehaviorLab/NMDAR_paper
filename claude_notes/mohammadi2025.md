# Mohammadi et al. 2025 — GLM-HMM-T Paper

**Full title**: Identifying the factors governing internal state switches during nonstationary sensory decision-making  
**Authors**: Zeinab Mohammadi, Zoe C. Ashwood, International Brain Laboratory & Jonathan W. Pillow  
**Journal**: Nature Communications (2026)  
**DOI**: https://doi.org/10.1038/s41467-025-66738-0  
**Relevance**: Methodological foundation for the GLM-HMM-T model used in the NMDAR paper

---

## Core Contribution

Develops **GLM-HMM-T** (GLM Hidden Markov Model with input-driven Transitions): extends the standard GLM-HMM (Ashwood et al. 2022) by replacing fixed transition probabilities with a **multinomial GLM** whose inputs are trial-by-trial behavioural covariates. This allows the model to identify *what drives* switches between internal states, not just detect them.

---

## Dataset

- **IBL Repeated Site dataset**: 123 mice performing a visual contrast discrimination task
- 37 mice selected for full analysis (strict behavioral criteria)
- Hundreds of thousands of trials total
- Task: report which side a Gabor patch appears on (6 contrasts per side + 0%, 6 blocks of ~90 trials with L/R prior probability blocks)

---

## Model Architecture

### 4 Hidden States (selected as parsimonious via cross-validation)
1. **Engaged-L** — stimulus-driven, slight leftward bias
2. **Engaged-R** — stimulus-driven, slight rightward bias
3. **Biased-L** — strong leftward bias, history-driven
4. **Biased-R** — strong rightward bias, history-driven

### Observation GLM (per state, predicts choice)
Covariates: stimulus contrast × side, past choice, bias term

### Transition GLM (predicts state-to-state switches)
Covariates:
- **Filtered previous choices** (exponentially weighted history)
- **Filtered previous stimuli** (exponentially weighted history)
- **Filtered previous rewards** (exponentially weighted history)
- **3 temporal basis vectors** (captures warm-up effect in first ~100 trials of session)

---

## Key Findings on State Transitions

1. **Past choices + past stimuli → left/right bias switches**: Animals switch between Biased-L ↔ Biased-R (and Engaged-L ↔ Engaged-R) driven by recent choice and stimulus history — consistent with a win-stay/lose-shift strategy
2. **Past rewards → engaged/disengaged switches**: More rewards → higher probability of *leaving* the Engaged states (disengagement), suggesting **satiety** as a driver of disengagement
3. Transition patterns are consistent across mice and explain much of the non-stationarity in behavior

---

## Model Comparison

- GLM-HMM-T outperforms standard GLM-HMM (fixed transitions) in held-out log-likelihood
- 4-state model selected over 2, 3, 5-state alternatives via 5-fold cross-validation
- GLM-T (single state, input-driven transitions) also outperforms plain GLM but not GLM-HMM-T

---

## Fitting Details

- **MAP estimation** with EM algorithm
- **5-fold cross-validation** for model selection
- States identified by posterior state probabilities (Viterbi decoding or soft assignments)

---

## Relevance to NMDAR Paper

The NMDAR paper applies this exact GLM-HMM-T framework to assess how NMDAR antagonist administration shifts the mouse between internal states. The NMDAR paper uses the same 4-state structure and transition covariates, and cites this paper in the Discussion section describing the GLM-HMM-T methodology.

---

## Citation

Mohammadi Z, Ashwood ZC, International Brain Laboratory, Pillow JW. Identifying the factors governing internal state switches during nonstationary sensory decision-making. *Nat Commun*. 2025. https://doi.org/10.1038/s41467-025-66738-0
