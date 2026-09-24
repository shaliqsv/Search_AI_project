# D4. Synthetic categories (settled in modelling Step 2)

Status: Accepted for Level 0 (was Provisional).

## Decision
Build the item co-visitation graph from the TRAINING window (days 0-9) only, embed items with truncated SVD (64 dimensions), cluster with k-means at K = 80 and name the clusters from the existing fictional taxonomy (80 leaves in 15 groups) by pairing clusters in dendrogram order of their centroids with leaves in tree order. Items with no co-visits (1,902 of 305,061) have no category. Categories are labelled synthetic everywhere.

## Why
K from 50 to 100 was swept with three seeds each. K = 80 has stability (ARI) 0.763 against 0.794 for the best K (90; noise 0.036 = the larger pairwise spread of the two, five seeds), a gap of 0.031, inside the noise; its largest cluster holds 3.9% of items and none is small. The taxonomy has exactly 80 leaves, and the classifier evaluation set is written against those names; another K would need a new taxonomy, which needs Claude access.

## Alternatives considered
Leiden community detection (69 communities at its most stable resolution, stability 0.828, largest community 25.6%, 50 smaller than 500 items): the number of communities is not controlled, so it cannot be paired with a fixed taxonomy. Opaque labels; item2vec embeddings; clustering on all weeks (leak).

## Known leak (D5)
The ranking query is the category of the session's held-out next click plus the prefix. It makes ranking inside the category easier than real search. Never present ranking results as evidence of real query understanding.
