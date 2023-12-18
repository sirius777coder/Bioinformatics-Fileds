# 量子化学 quantum chemistry

[toc]

## QM basics

### Dihedral angles

- Def : A dihedral angle is the angle between two intersecting plans or half-plans. 

- 范围 （注意边界条件的取值）
  - 在高中数学中，计算二面角得到$\cos \phi$ , **默认二面角的范围是 $[0,\pi)$** 
  
  - 在有机化学、空间化学(stereochemistry)，**生物化学中；我们定义二面角的范围是$(-\pi, \pi]$**
    - **定义**：在化学领域通常是有一系列顺序标记的点$\bold{r_1},\bold{r_2},\bold{r_3},\bold{r_4}$ , 此时二面角定义为$\bold{r_1},\bold{r_2},\bold{r_3}$  组成的平面和$\bold{r_2},\bold{r_3},\bold{r_4}$  组成的平面的夹角。$\bold{r}_1$与$\bold{r}_4$处于同一侧称为顺式构象cis，处于相反侧为trans，对于蛋白质肽键形成的二面角绝大部分都是反式构象，即二面角处于+-180度之间 (180度再按照顺时针旋转就为-179度, - 178度)
    
    - **二面角旋转**：我们旋转二面角时是从$\bold{r}_4$ 看向$\bold{r}_1$，顺式构象cis 为0度，**顺时针**绕着$\bold{r}_2, \bold{r}_3$ 组成的轴进行旋转，旋转到180度后再进行旋转为-180度。**旋转时$\bold{r_2},\bold{r_3},\bold{r_4}$ 三个点保持不动，唯一发生变化的是$\bold{r}_1$以及和$\bold{r}_2$ 相连的我们不敢兴趣的点** 
    
    - **蛋白质肽键对应的二面角**： $\omega$ 角度连接的$\bold{r}_1=CA_i,\bold{r}_2=C_i,\bold{r}_3=N_{i+1},\bold{r}_4=CA_{i+1}$  按照45度进行扫描`0,45,90,135,180,-135,-90,-45,0` ，我们只改变$CA_i$的位置以及与$Ci$连接的羧基氧原子
    
      - $\phi$角度对应$\bold{r}_1=C_{i-1},\bold{r}_2=N_i,\bold{r}_3=CA_{i},\bold{r}_4=C_{i}$ (r1为第i-1个氨基酸，r234为第i个氨基酸)
      - $\psi$ 角度对应 $\bold{r}_1=N_{i},\bold{r}_2=CA_i,\bold{r}_3=C_{i},\bold{r}_4=N_{i+1}$ (r123为第i个氨基酸，r4为第i+1个氨基酸)
    
    - E.g. 正丁烷n-butane对应的不同torsion angle
    
      - 将$(-\pi,\pi]$ 进行区间划分 
        - 0 到 +- 30 称为 synperiplanar 或 **cis**
        - +-30 到 +- 90称为 synclinal  或**gauche** , 或者称为顺时针60度为gauche + ，逆时针60度为gauche - 
        - +-90 到 +- 150 称为 anticlinal
        - +-150 到 +- 180 称为 antiperiplanar 或 **trans**
    
      ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Dihedral_angles_of_Butane.svg/1920px-Dihedral_angles_of_Butane.svg.png)
    
    > E.g. AF2 SI.  " As we are only predicting heavy atoms, the extra backbone rigid groups *ω* and *φ* do not contain atoms, but the corresponding frames contribute to the FAPE loss for alignment to the ground truth (like all other frames)."
    
    
  
  - 如何计算蛋白质的二面角
  
  ```python
   def _dihedrals(X, eps=1e-7):
      # First 3 coordinates are N, CA, C
      X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)
  
      # Shifted slices of unit vectors
      dX = X[:,1:,:] - X[:,:-1,:] # 3N-1 unit vector
      U = F.normalize(dX, dim=-1)
      u_2 = U[:,:-2,:]
      u_1 = U[:,1:-1,:]
      u_0 = U[:,2:,:]
      # Backbone normals
      n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
      n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
  
      # Angle between normals
      cosD = (n_2 * n_1).sum(-1)
      cosD = torch.clamp(cosD, -1+eps, 1-eps)
      D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
  
      # This scheme will remove phi[0], psi[-1], omega[-1]
      D = F.pad(D, (1,2), 'constant', 0)
      D = D.view((D.size(0), int(D.size(1)/3), 3))
      phi, psi, omega = torch.unbind(D,-1)
  
      # Lift angle representations to the circle
      D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
      return D_features
  ```
  
  



### hessian matrix



对于过渡态来说，过渡态是IRC上的最大值，其他方向上的最小值，因此标准的TS对应的Hessian矩阵应该 **有且只有一个负的特征值** 表面在该路径上TC是极大值
