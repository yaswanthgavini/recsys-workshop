# Participant setup (do this BEFORE the workshop, ~10 minutes)

You need **Python 3.9 or newer** and a laptop with ~1 GB free space. Pick ONE route.

## Route A — your own laptop (recommended)
```bash
python3 -m venv recsys && source recsys/bin/activate        # Windows: recsys\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Check it works (paste in a notebook cell):
```python
import numpy, pandas, sklearn, matplotlib, surprise
print(numpy.__version__, pandas.__version__, surprise.__version__)   # surprise 1.1.x, numpy 1.x
```

### If `scikit-surprise` fails to install
| Symptom | Fix |
|---|---|
| build error / `numpy.core.multiarray failed to import` | `pip install "numpy<2" scikit-surprise` |
| Windows: "Microsoft Visual C++ 14.0 required" | `conda install -c conda-forge scikit-surprise` (avoids compiling) |
| still stuck | use **Route B**. Labs 1 and 2 do **not** need Surprise; only Lab 3 (14:15) does. |

## Route B — Google Colab (no installation)
1. Upload the notebook (`File → Upload notebook`).
2. First cell: `!pip install -q "numpy<2" scikit-surprise` then **Runtime → Restart session**.
3. The notebooks download the dataset themselves if it is missing.

## Data
The notebooks look for `data/ml-latest-small/` and download it if absent (about 1 MB):
MovieLens *latest-small*, GroupLens Research, University of Minnesota — for research and education; please do not redistribute.
Citation: F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015.

## Which notebook when?
| Time | Notebook | Needs |
|---|---|---|
| 09:45 | `Lab1_EDA.ipynb` | pandas, matplotlib |
| 11:45 | `Lab2_Collaborative_Filtering.ipynb` | + scikit-learn; needs Lab 1's saved split (run Lab 1 top to bottom if you missed it, ~30 s) |
| 14:15 | `Lab3_Matrix_Factorization_SVD.ipynb` | + **scikit-surprise** |

Use the files in `labs/notebooks/student/`. Instructors keep the solved versions in `labs/notebooks/instructor/`.
