---
title: 'Performance of a Reinforcement Learning Model Fitting Toolkit on Simulated Data from the Two-Step Task'
author: Abraham Nunes
date: August 18, 2016
---

## Simulated Subjects

Simulated subjects included the following parameters:

- Model-based/model-free weight $\omega$
- Learning rate $\alpha$
- Inverse temperature $\beta$
- Eligibility trace $\lambda$
- Reward sensitivity $\rho$

## Generative Models

Data were generated from models with the following parameters:

------------------------------------------------------
1. $\lbrace \alpha, \beta, \omega \rbrace$
2. $\lbrace \alpha, \beta, \rho, \omega \rbrace$
3. $\lbrace \alpha, \beta, \lambda, \omega \rbrace$
4. $\lbrace \alpha, \beta, \lambda, \rho, \omega \rbrace$
5. $\lbrace \alpha, \beta, \omega = 0 \rbrace$
6. $\lbrace \alpha, \beta, \rho, \omega = 0 \rbrace$
7. $\lbrace \alpha, \beta, \lambda, \omega = 0 \rbrace$
8. $\lbrace \alpha, \beta, \lambda, \rho, \omega = 0 \rbrace$
9. $\lbrace \alpha, \beta, \omega = 1 \rbrace$
10. $\lbrace \alpha, \beta, \rho, \omega = 1 \rbrace$
11. $\lbrace \alpha, \beta, \lambda, \omega = 1 \rbrace$
12. $\lbrace \alpha, \beta, \lambda, \rho, \omega = 1 \rbrace$
------------------------------------------------------

## Models to fit

Models used to generate the data were mirrored for the fitting process, with the addition of a model that has a single parameter(inverse temperature), and simply makes scaled random choices at each step.

 
