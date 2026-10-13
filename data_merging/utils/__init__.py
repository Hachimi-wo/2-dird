"""Shared helpers used by the data-merging scripts.

The two pipeline scripts live in ``data_merging/`` and import from this
package:

* ``01_curate_sources.py`` — filter each public source, fetch SMILES, run MEHC
* ``02_build_dataset.py`` — merge, label, validate negatives, write the cleaned set

Modules
-------
paths
    Absolute locations under ``dird/data`` (raw, intermediate, curated, cleaned).
mehc
    Thin wrapper around ``mehc_curation`` so every source uses the same SMILES
    standardization.
pubchem
    PubChem PUG REST (name → CID / SMILES) and PUG-View / assay scoring used
    to keep only unlabeled drugs with no pulmonary-ADR evidence as negatives.
"""
