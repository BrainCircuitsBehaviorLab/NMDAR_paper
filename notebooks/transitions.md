## Full directed transition model

The full directed model is:

$$
\eta_{ij}(t) = b_{ij} + u_{t+1}^{\top} W_{ij}
$$

$$
P(z_{t+1}=j \mid z_t=i, u_{t+1})
=
\frac{
\exp\left(b_{ij} + u_{t+1}^{\top} W_{ij}\right)
}{
\sum_{\ell=1}^{K}
\exp\left(b_{i\ell} + u_{t+1}^{\top} W_{i\ell}\right)
}
$$

However, this full form is not identifiable, because adding the same value to every logit in a source row does not change the softmax probabilities.

For any source state $i$, if:

$$
b'_{ij} = b_{ij} + a_i
$$

and

$$
W'_{ij} = W_{ij} + c_i
$$

for all target states $j$, then the transition probabilities are unchanged.

Therefore, one logit per source row must be used as a reference.

---

## What we freeze

We use the self-transition as the reference logit.

For every source state $i$, we freeze:

$$
\eta_{ii}(t) = 0
$$

So the transition probabilities are:

$$
P(z_{t+1}=i \mid z_t=i, u_{t+1})
=
\frac{
1
}{
1 + \sum_{\ell \neq i}
\exp\left(\beta_{i\ell} + u_{t+1}^{\top} V_{i\ell}\right)
}
$$

$$
P(z_{t+1}=j \mid z_t=i, u_{t+1})
=
\frac{
\exp\left(\beta_{ij} + u_{t+1}^{\top} V_{ij}\right)
}{
1 + \sum_{\ell \neq i}
\exp\left(\beta_{i\ell} + u_{t+1}^{\top} V_{i\ell}\right)
},
\qquad j \neq i
$$

---

## For $K = 2$

$$
\eta_{EE}(t) = 0
$$

$$
\eta_{DD}(t) = 0
$$

and fit:

$$
\eta_{ED}(t) = \beta_{ED} + u_{t+1}^{\top} V_{ED}
$$

$$
\eta_{DE}(t) = \beta_{DE} + u_{t+1}^{\top} V_{DE}
$$

Therefore:

$$
P(E \to D)
=
\frac{
\exp\left(\beta_{ED} + u_{t+1}^{\top} V_{ED}\right)
}{
1 + \exp\left(\beta_{ED} + u_{t+1}^{\top} V_{ED}\right)
}
$$

$$
P(E \to E)
=
\frac{
1
}{
1 + \exp\left(\beta_{ED} + u_{t+1}^{\top} V_{ED}\right)
}
$$

$$
P(D \to E)
=
\frac{
\exp\left(\beta_{DE} + u_{t+1}^{\top} V_{DE}\right)
}{
1 + \exp\left(\beta_{DE} + u_{t+1}^{\top} V_{DE}\right)
}
$$

$$
P(D \to D)
=
\frac{
1
}{
1 + \exp\left(\beta_{DE} + u_{t+1}^{\top} V_{DE}\right)
}
$$
---

## Paper _C. elegans_ formulation

$$
P(z_{t+1}=j \mid z_t=i, C)
=
\frac{
\exp\left(K_{ij} \cdot C + b_i\right)
}{
Z_i(C)
}
$$

$$
\eta_{ii}(t) = b_i
$$

$$
\eta_{ij}(t) = K_{ij} \cdot C_t,
\qquad j \neq i
$$

$$
b_{ii} = b_i
$$

$$
b_{ij} = 0,
\qquad j \neq i
$$

equivalent to our current model if:
$$
\beta_{ij}=-b_i \qquad
V_{ij}=W_{ij}
$$

---

## Mohammadi formulation


$$
P(z_t=j \mid z_{t-1}=i, x_t^{tr})
\propto
\exp\left(B_{ij} + (w_j^{tr})^\top x_t^{tr}\right)
$$

Normalized:

$$
P(z_t=j \mid z_{t-1}=i, x_t^{tr})
=
\frac{
\exp\left(B_{ij} + (w_j^{tr})^\top x_t^{tr}\right)
}{
\sum_{\ell=1}^{K}
\exp\left(B_{i\ell} + (w_\ell^{tr})^\top x_t^{tr}\right)
}
$$

$$
W_{E \to D} = W_{D \to D} = w_D
$$

and

$$
W_{E \to E} = W_{D \to E} = w_E
$$

