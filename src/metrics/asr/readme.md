
# Liver zonation anatomical landmark annotations 
PA (Portal Area) and CV (Central Vein) boundaries are optional Polygon
shapes manually annotated by a pathologist on the morphology image using
QuPath.  They encode the two anatomical landmarks of the hepatic lobule:

  PA — periportal zone; surrounds the portal triad (hepatic artery,
       portal vein, bile duct); Zone 1 gene expression peaks here.
  CV — centrilobular zone; surrounds the central vein; Zone 3 gene
       expression peaks here.

Coordinate system: shapes are stored in µm (intrinsic/element space)
with a scale transform of 1/pixel_size (≈ 4.706 for 0.2125 µm/px
Xenium data) mapping to the global (pixel) coordinate system — exactly
the same convention used for cell_boundaries and nucleus_boundaries.

They are consumed downstream by the Assignment Specificity Ratio (ASR)
metric (src/metrics/asr/) to measure transcript mis-assignment in liver
segmentation benchmarks by comparing zonal expression gradients between
hepatocytes and sinusoidal endothelial cells.

IMPORTANT:
  These shapes are ONLY present for liver datasets that have been
  prepared with manual annotations.  The standard mouse-brain
  test dataset (resources_test/…/mouse_brain_combined) does NOT
  contain pa_boundaries / cv_boundaries.  The ASR metric handles this
  gracefully (returns NaN), but it means ASR cannot be meaningfully
  compared across liver and non-liver datasets in the same benchmark
  run. 

