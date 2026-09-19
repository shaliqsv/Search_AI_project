# OTTO dataset license and terms

Checked 2026-09-19.

## Sources
- Dataset repository: https://github.com/otto-de/recsys-dataset
  - Dataset: CC BY 4.0. Code: MIT.
  - CC BY 4.0 allows copying, sharing and adapting (including derived samples) for any purpose, if you give credit, link the license and say what you changed.
- Kaggle competition: https://www.kaggle.com/competitions/otto-recommender-system
  - The rules and data pages could not be read automatically (they need a logged-in browser). NOT VERIFIED. Read the Rules and Data tabs by hand before publishing anything derived.

## Answers
| Question | Answer | Basis |
|---|---|---|
| Can derived samples be committed to a public repo? | Unclear | Allowed by CC BY 4.0 with attribution, but the Kaggle rules are unverified. Conservative rule: do not commit |
| Can trained models be published? | Unclear | Same reason. Models are adaptations under CC BY 4.0 with attribution, but unverified against Kaggle |
| Can metrics, plots and screenshots be published? | Yes | They do not contain the data itself. Give attribution |

## Rules for this repo
- `data/` is gitignored. Do not commit OTTO data or derived samples until the two "Unclear" rows are settled.
- Anywhere results are published, credit OTTO and link CC BY 4.0, and state that the data was sampled and modified.
