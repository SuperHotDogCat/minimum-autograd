# minimum-autograd
## 扱うNeural Network
$y = xW_1$
$z = \text{ReLU}(y)$
$w = zW_2$
$p = \text{Sigmoid}(w)$
$L = -\text{mean}(y_{true}\log{(p)}+(1-y_{true})\log{(1-p)})$
## 計算されるべきgradient
$\partial_p L = -\dfrac{y_{true}}{p}+\dfrac{1-y_{true}}{1-p} = \dfrac{p-y_{true}}{p(1-p)}$
$\partial_w L = \partial_p L \odot \partial_w p = \partial_p L \odot p(1-p)$
$\partial_z L = \partial_w L * \partial_z w = \partial_w L * W_2^{T}$
$\partial_{W_2} L = \partial_{W_2} w * \partial_w L = z^T * \partial_w L$
$\partial_y L = \partial_z L \odot \partial_y z = \partial_z L \odot \text{where}(y > 0, 1, 0)$

n m m k -> n k