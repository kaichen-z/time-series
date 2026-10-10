# Dr-CiK full-199 split

`split.json` records the split used for the full Dr-CiK experiment:

- 199 public-label tasks form the training universe.
- 80 private-label tasks form the hidden test universe.
- The 199 training tasks are divided into three deterministic, entity-disjoint
  internal folds of 67, 66, and 66 tasks using seed 7.

The manifest contains task IDs, fold entities, and aggregate split metadata.
It contains no target values, evidence labels, or hidden-test labels.

File SHA-256:
`9a89fa497f71bf19f04e3e19e78c2c62541a44a73730affe6dfd513d222b7f7b`
