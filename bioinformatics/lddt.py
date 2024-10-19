import biotite.structure as struc
import biotite.structure.io as strucio
import os
import numpy as np

def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
  """Measure (approximate) lDDT for a batch of coordinates.

  lDDT reference:
  Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
  superposition-free score for comparing protein structures and models using
  distance difference tests. Bioinformatics 29, 2722-2728 (2013).

  lDDT is a measure of the difference between the true distance matrix and the
  distance matrix of the predicted points.  The difference is computed only on
  points closer than cutoff *in the true structure*.

  This function does not compute the exact lDDT value that the original paper
  describes because it does not include terms for physical feasibility
  (e.g. bond length violations). Therefore this is only an approximate
  lDDT score.

  Args:
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    true_points_mask: (batch, length, 1) binary-valued float array.  This mask
      should be 1 for points that exist in the true points.
    cutoff: Maximum distance for a pair of points to be included
    per_residue: If true, return score for each residue.  Note that the overall
      lDDT is not exactly the mean of the per_residue lDDT's because some
      residues have more contacts than others.

  Returns:
    An (approximate, see above) lDDT score in the range 0-1.
  """

  assert len(predicted_points.shape) == 3
  assert predicted_points.shape[-1] == 3
  assert true_points_mask.shape[-1] == 1
  assert len(true_points_mask.shape) == 3

  # Compute true and predicted distance matrices.
  dmat_true = np.sqrt(1e-10 + np.sum(
      (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

  dmat_predicted = np.sqrt(1e-10 + np.sum(
      (predicted_points[:, :, None] -
       predicted_points[:, None, :])**2, axis=-1))

  dists_to_score = (
      (dmat_true < cutoff).astype(np.float32) * true_points_mask *
      np.transpose(true_points_mask, [0, 2, 1]) *
      (1. - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
  )

  # Shift unscored distances to be far away.
  dist_l1 = np.abs(dmat_true - dmat_predicted)

  # True lDDT uses a number of fixed bins.
  # We ignore the physical plausibility correction to lDDT, though.
  score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                  (dist_l1 < 1.0).astype(np.float32) +
                  (dist_l1 < 2.0).astype(np.float32) +
                  (dist_l1 < 4.0).astype(np.float32))

  # Normalize over the appropriate axes.
  reduce_axes = (-1,) if per_residue else (-2, -1)
  norm = 1. / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
  score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))

  return score

def lddt_ca(native,predict,mask=None,**kwargs):
  """
  ZB modofied from chenyinghui's code
  """
  native = strucio.load_structure(native)
  predict = strucio.load_structure(predict)
  
  native_CA = native[native.atom_name == "CA"].coord
  native_CA = np.expand_dims(native_CA, axis=0)
  predict_CA = predict[predict.atom_name == "CA"].coord
  predict_CA = np.expand_dims(predict_CA, axis=0)
  if mask is None:
    mask = np.ones((1, len(native_CA), 1))
  return lddt(predict_CA, native_CA, mask, **kwargs)

esmfold_lddt_list = []
for pdb in os.listdir("/xcfhome/bozhang/data/TMPNN_format/test"):
    if pdb.endswith(".pdb"):
        esmfold_lddt_list.append(lddt_ca(f"/xcfhome/bozhang/data/TMPNN_format/test/{pdb}", f"/xcfhome/bozhang/NeMO_ablation/native_seqeuence_refold/esmfold/{pdb}").item())