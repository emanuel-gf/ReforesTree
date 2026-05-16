# ReforesTree — Agroforestry Drone Imagery for Above Ground Biomass Estimation

The ReforesTree dataset is a high-resolution RGB drone imagery dataset collected across 6 agroforestry sites in Ecuador, designed to support machine learning research on tropical reforestation inventory. Each image tile is accompanied by field-measured tree-level annotations, including species identity and estimated Above Ground Biomass (AGB).

---

## The Ecological Context

Tropical agroforestry systems are among the most complex and biodiverse land-use types in the world. Unlike monoculture plantations, they combine multiple tree species — timber, fruit, and shade trees — alongside crops such as cacao and banana. This diversity makes large-scale biomass monitoring both critical and challenging.

**Above Ground Biomass (AGB)** refers to the total dry mass of living material above the soil surface (trunks, branches, leaves, and bark). It is the primary indicator used to estimate the carbon stored in a forest or agroforestry system, and therefore a key metric for:

- Carbon credit verification under REDD+ and voluntary carbon markets
- National greenhouse gas inventories
- Monitoring reforestation project outcomes over time

Traditionally, AGB is estimated through **forest inventory field campaigns**: trained ecologists walk every plot, identify each tree by species, and measure its diameter at breast height (DBH, measured at 1.3 m from the ground) and sometimes its total height. These physical measurements are then fed into **allometric equations** — species-specific mathematical models derived from decades of dendrology research — to compute biomass in kilograms or megagrams (Mg). For example, a simplified pantropical equation takes the form:

$$\text{AGB} = 0.0673 \times (\rho \cdot D^2 \cdot H)^{0.976}$$

where $\rho$ is wood density (g/cm³), $D$ is DBH (cm), and $H$ is tree height (m).

This process is accurate but labor-intensive, expensive, and impossible to scale across thousands of hectares. The ReforesTree dataset was created to bridge this gap — pairing drone imagery with field-measured AGB so that machine learning models can learn to infer biomass directly from aerial photographs.

---

## Data Collection

### Drone Imagery

Each of the 6 sites was surveyed using a **DJI Phantom 4 Pro** UAV flying at low altitude, producing orthomosaic images at approximately **2 cm/pixel** ground sampling distance. At this resolution, individual tree crowns are clearly distinguishable, and structural features such as crown shape, texture, and shadow are well preserved.

The raw orthomosaics are large raster files (tens of thousands of pixels per side). For practical use, they were sliced into **4,000 × 4,000 pixel tiles**, each covering roughly **80 × 80 metres** of ground area. The dataset contains **100 such tiles** across the 6 sites.

### Field Inventory

In parallel with the drone surveys, field teams conducted a tree-by-tree inventory of each site. For every tree within the survey area, the following parameters were recorded:

- **Species name** (both common and scientific)
- **Diameter at breast height (DBH)** — measured with a diameter tape at 1.3 m height
- **GPS coordinates** — recorded with a handheld device at the base of each tree
- **Species group** — a coarse classification (banana, cacao, citrus, fruit, timber, other)

Each drone imagery patch therefore contains sample field and hand-measured tree parameters — such as breast height and species identification — which led to the calculation of biomass based on aspects and equations species-related, derived from dendrology study. Thus, this dataset contains, for each tree described within the inventory, an estimation of above ground biomass.

The field data is stored in `field_data.csv`, with one row per inventoried tree.

### Bounding Box Annotation

To link the field measurements to the drone imagery, bounding boxes around individual tree crowns were generated using [DeepForest](https://deepforest.readthedocs.io/en/stable/), a deep learning model trained for tree crown detection. These automatically generated boxes were then **manually cleaned** by the dataset authors to remove false detections and correct misaligned boxes.

Each bounding box was matched to a field measurement record using GPS proximity — the box is assigned to the field-measured tree whose GPS coordinate falls closest to the box centroid. This matching process is the primary source of label noise in the dataset: GPS accuracy in a forest canopy environment can be off by several metres, and not every visible tree crown has a corresponding field measurement.

---

## Dataset Structure

```
reforesTree/
├── tiles/                          # 4000×4000 px RGB PNG tiles, organized by site
│   ├── Carlos Vera Arteaga RGB/
│   ├── Carlos Vera Guevara RGB/
│   ├── Carlos Vera Morejón RGB/
│   ├── Flora Pluas RGB/
│   ├── Nestor Lema RGB/
│   └── Cardenal RGB/
└── mapping/
    └── final_dataset.csv           # one row per annotated tree
```

### `mapping/final_dataset.csv` — Key Columns

| Column | Description |
|---|---|
| `img_path` | Filename of the tile this tree belongs to |
| `xmin`, `ymin`, `xmax`, `ymax` | Bounding box in pixel coordinates (XYXY format) |
| `name` | Fine-grained species name |
| `group` | Coarse species group (banana, cacao, citrus, fruit, timber, other) |
| `AGB` | Estimated above ground biomass in **kilograms**, computed from allometric equations applied to DBH field measurements |

### AGB Distribution

AGB values are right-skewed: most trees are small-to-medium cacao or banana plants with AGB below 50 kg, while a minority of timber trees can reach several hundred kilograms. The dataset-wide statistics are approximately:

- Mean AGB: ~13 kg
- Std AGB: ~54.5 kg
- Range: < 1 kg (seedlings) to > 500 kg (large timber)

This skew is important for model design — raw AGB values should be normalized before training a regression model.

---

## The 6 Agroforestry Sites

All sites are located in coastal Ecuador, a region with a tropical climate and a long tradition of smallholder agroforestry. The sites differ in species composition, canopy density, and management intensity.

| Site | Primary species |
|---|---|
| Carlos Vera Arteaga | Cacao, timber, fruit |
| Carlos Vera Guevara | Cacao, banana, citrus |
| Carlos Vera Morejón | Mixed agroforestry |
| Flora Pluas | Cacao, banana |
| Nestor Lema | Timber, fruit |
| Cardenal | Mixed agroforestry |

> Five tiles from Carlos Vera Guevara and Flora Pluas are excluded from training due to a known GPS mismatch ([issue #6](https://github.com/gyrrei/ReforesTree/issues/6)) where bounding boxes do not align to the correct image region.

---

## Loading the Dataset

### Via TorchGeo (recommended)

```python
from torchgeo.datasets import ReforesTree

ds = ReforesTree(root="data/reforestree/", download=True, checksum=True)
sample = ds[0]
# sample keys: 'image', 'bbox_xyxy', 'label', 'agb', ...
```

### On Kaggle

The dataset is available as a Kaggle dataset. Add it to your notebook via:  
**+ Add Data → Search `reforestree-dataset`**

The root path will be `/kaggle/input/reforestree-dataset/`.

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{reiersen2022reforestree,
  title     = {ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Deep Learning and Aerial Imagery},
  author    = {Reiersen, Gyri and Dao, David and Lütjens, Björn and Klemmer, Konstantin and Amara, Kenza and Steinegger, Attila and Zhang, Ce and Zhu, Xiaoxiang},
  journal   = {arXiv preprint arXiv:2201.11192},
  year      = {2022}
}
```

Paper: [arxiv.org/abs/2201.11192](https://arxiv.org/abs/2201.11192)  
Original repository: [github.com/gyrrei/ReforesTree](https://github.com/gyrrei/ReforesTree)  
Zenodo archive: [zenodo.org/record/6813783](https://zenodo.org/record/6813783)
