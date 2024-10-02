## Equation
helmholtz equation 2D
$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + k^2u=0
$$
boundary conditions, $[0,L_x]\times[0,L_y]$
$$
u(x,0)=0 \quad
u(x, L_y)=\sin\left(\frac{2\pi x}{L_x}\right)
$$
$$
u(0,y)=0 \quad
u(L_x, y)=0
$$

## Solution
$$
    u(x,y)=\frac{1}{\sinh(\mu_2L_y)}\sin\left(\frac{2\pi x}{L_x}\right)\sinh(\mu_2y)
$$
$$
\mu_2=\sqrt{\frac{4\pi^2}{L_x^2}-k^2}
$$
for $L_x=L_y=k=1$ we have $\mu_2=\sqrt{4\pi^2-1}$ and solution:
$$
    u(x,y)=\frac{1}{\sinh(\sqrt{4\pi^2-1})}\sin(2\pi x)\sinh(y\sqrt{4\pi^2-1})
$$

## PINN solution
used $\tanh$ activation function 8 hidden layers and 8 nodes each hidden layer. optimized with adam optimizer.

solving for points $D\subset [0,1]\times[0,1]$
$$
    D=\{(x_1,y_1),(x_2,y_2),(x_3,y_3),...\}
$$
PINN takes a point from $D$ and gives prediction $\hat{u}$
$$
\hat{u}(x,y)=PINN(x,y)
$$
Minimizing Error for equation
$$
E_{equation}=\frac{1}{|D|}\sum_{p\in D}(\hat{u}_{xx}(p)+\hat{u}_{yy}(p)+k^2\hat{u}(p))^2
$$
and for boundary points $\partial D$
$$
E_{boundary}=\frac{1}{|\partial D|}\sum_{p\in \partial D}(\hat{u}(p)-0)^2
$$