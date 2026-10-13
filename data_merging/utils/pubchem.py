"""PubChem helpers: resolve structures and score unlabeled drugs.

Used in two places
------------------
Phase 1 (``01_curate_sources.py``)
    Look up CID / Canonical SMILES so each curated table has a structure
    before MEHC.

Phase 2 (``02_build_dataset.py``)
    Unlabeled rows (all four ADR labels = 0) are not trusted as negatives.
    PUG-View + assay summary are searched for pulmonary-ADR language. Only
    compounds with a combined score of 0 are kept as negatives.

Rate limiting
-------------
PubChem asks for a descriptive User-Agent and modest request rates. Callers
sleep between rows (``SLEEP``); ``smiles_batch`` also backs off on HTTP 503.
Network errors return empty/None so a single failed lookup does not abort
a multi-hour curation run.
"""

import json
import re
import time
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (DIRD data merging; research use)",
}
TIMEOUT = 60
# Default pause between per-compound PUG-View / assay calls in phase 2.
SLEEP = 0.4
MAX_CONSECUTIVE_RATE_LIMITS = 3


class PubChemRateLimitError(RuntimeError):
    """Raised when PubChem keeps returning 429 and the run should resume later."""

# Pulmonary terms aligned with the four labels. Matched as substrings on
# lowercased JSON/text, so keep phrases specific enough to avoid unrelated hits
# (e.g. "pe" is included because PubChem often abbreviates pulmonary embolism,
# but it is only counted inside adverse-event sections — see ADVERSE_TERMS).
KEYWORDS = [
    # ILD / fibrosis / pneumonitis
    "interstitial lung disease", "ild", "pulmonary fibrosis", "pneumonitis",
    "interstitial pneumonia", "fibrosing alveolitis",
    "usual interstitial pneumonia", "desquamative interstitial pneumonia",
    "nonspecific interstitial pneumonia", "cryptogenic organizing pneumonia",
    "hypersensitivity pneumonitis", "sarcoidosis",
    "idiopathic pulmonary fibrosis", "ipf",
    # pleural
    "pleural effusion", "pleurisy", "pleuritis", "pneumothorax",
    "pleural thickening", "pleural plaque", "mesothelioma",
    # pulmonary edema
    "pulmonary edema", "pulmonary oedema", "acute pulmonary edema",
    "noncardiogenic pulmonary edema", "fluid accumulation", "hydrothorax",
    "lung edema", "capillary leak",
    # embolism / thrombosis
    "pulmonary embolism", "thromboembolism", "venous thromboembolism",
    "pulmonary thromboembolism", "deep vein thrombosis", "dvt", "pe",
]

# A keyword hit only counts if it appears under an adverse/toxicity heading
# (or in an assay whose text mentions these terms). Otherwise disease-indication
# text (e.g. a drug used to treat fibrosis) would inflate the negative score.
ADVERSE_TERMS = [
    "adverse", "side effect", "toxicity", "safety", "hazard", "warning",
    "drug induced", "pulmonary toxicity", "respiratory toxicity", "lung toxicity",
]


def _get(url, timeout=TIMEOUT):
    """GET with the project User-Agent. Callers handle status codes and errors."""
    return requests.get(url, headers=HEADERS, timeout=timeout)


def _retry_delay(response, attempt, base=2.0):
    """Return a server-aware exponential backoff delay in seconds."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), base)
        except ValueError:
            pass
    if response.status_code == 429:
        base = max(base, 30.0)
    elif response.status_code >= 500:
        base = max(base, 5.0)
    return base * (2 ** attempt)


def is_valid_smiles(value):
    """Return True only for a non-empty SMILES string accepted by RDKit."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.casefold() in {"na", "nan", "none", "notfound"}:
        return False
    if text.casefold().startswith(("<!doctype", "<html")):
        return False
    return Chem.MolFromSmiles(text) is not None


def _smiles_from_url(url, empty, retries=4, delay=2.0):
    """Fetch and validate one PubChem SMILES response with exponential backoff."""
    for attempt in range(retries):
        response = None
        try:
            response = _get(url)
            if response.status_code == 404:
                return empty
            content_type = response.headers.get("Content-Type", "").casefold()
            text = response.text.strip()
            if (
                response.status_code == 200
                and "html" not in content_type
                and is_valid_smiles(text)
            ):
                return text
            if (
                response.status_code == 429
                and attempt + 1 >= MAX_CONSECUTIVE_RATE_LIMITS
            ):
                raise PubChemRateLimitError(
                    "PubChem returned HTTP 429 repeatedly; resume from the checkpoint later."
                )
        except requests.RequestException:
            pass
        if attempt + 1 < retries:
            wait = (
                _retry_delay(response, attempt, base=delay)
                if response is not None
                else delay * (2 ** attempt)
            )
            time.sleep(wait)
    return empty


def cid_from_name(name):
    """Resolve a drug name to the first PubChem CID, or None if not found.

    PubChem may return several CIDs (mixtures, salts). Taking the first is the
    usual PUG convention and matches how the rest of the pipeline stores a
    single ``PubChem_ID`` per row.
    """
    if not name:
        return None
    try:
        r = _get(f"{PUBCHEM}/name/{quote(str(name))}/cids/JSON")
        if r.status_code == 200:
            cids = r.json().get("IdentifierList", {}).get("CID") or []
            if cids:
                return cids[0]
    except Exception:
        return None
    return None


def smiles_from_cid(cid, retries=4, delay=2.0):
    """Canonical SMILES for one CID. Returns ``""`` on miss so callers can fill later."""
    url = f"{PUBCHEM}/cid/{cid}/property/CanonicalSMILES/TXT"
    return _smiles_from_url(url, "", retries=retries, delay=delay)


def smiles_from_name(name, retries=4, delay=2.0):
    """Canonical SMILES by name (used for CTD, which often lacks a CID).

    Returns None (not ``""``) so the CTD loop can try name variants until one
    succeeds.
    """
    url = f"{PUBCHEM}/name/{quote(str(name))}/property/CanonicalSMILES/TXT"
    return _smiles_from_url(url, None, retries=retries, delay=delay)


def smiles_batch(cids, checkpoint=None, batch=5, retries=6, delay=1.0):
    """Map many CIDs to Canonical SMILES with checkpointing and retries.

    Parameters
    ----------
    cids : sequence
        PubChem compound IDs (int or str). Already-fetched IDs in ``checkpoint``
        are skipped.
    checkpoint : path-like or None
        CSV with columns ``CID, SMILES``. Written after every batch so a
        crashed ADReCS run can resume.
    batch : int
        CIDs per PUG property request. Small on purpose: large comma-lists
        are more likely to 503.
    retries : int
        Attempts per batch before falling back to one CID at a time.
    delay : float
        Pause after each batch, successful or not.

    Returns
    -------
    dict
        ``{str(cid): smiles}``. Missing CIDs are simply absent; callers treat
        that as empty SMILES.
    """
    mapping = {}
    done = set()
    if checkpoint is not None and checkpoint.is_file():
        cached = pd.read_csv(checkpoint)
        cached["SMILES"] = cached["SMILES"].fillna("")
        mapping = {
            str(cid): smiles
            for cid, smiles in zip(cached["CID"], cached["SMILES"])
            if is_valid_smiles(smiles)
        }
        done = set(mapping)

    for i in range(0, len(cids), batch):
        chunk = [c for c in cids[i:i + batch] if str(c) not in done]
        if not chunk:
            continue
        cid_str = ",".join(str(c) for c in chunk)
        url = f"{PUBCHEM}/cid/{cid_str}/property/CanonicalSMILES,ConnectivitySMILES/JSON"
        ok = False
        for attempt in range(retries):
            try:
                r = _get(url, timeout=90)
                if r.status_code == 200:
                    props = (r.json().get("PropertyTable") or {}).get("Properties") or []
                    for prop in props:
                        # Prefer CanonicalSMILES; ConnectivitySMILES is a fallback
                        # when PubChem has no stereo-aware canonical form.
                        smiles = prop.get(
                            "CanonicalSMILES", prop.get("ConnectivitySMILES", "")
                        )
                        if is_valid_smiles(smiles):
                            mapping[str(prop["CID"])] = smiles
                    ok = True
                    break
                if r.status_code == 404:
                    # None of these CIDs exist; do not retry.
                    break
                if (
                    r.status_code == 429
                    and attempt + 1 >= MAX_CONSECUTIVE_RATE_LIMITS
                ):
                    if checkpoint is not None:
                        pd.DataFrame(
                            mapping.items(), columns=["CID", "SMILES"]
                        ).to_csv(checkpoint, index=False)
                    raise PubChemRateLimitError(
                        "PubChem returned HTTP 429 repeatedly; resume from the checkpoint later."
                    )
                time.sleep(_retry_delay(r, attempt, base=5.0))
            except PubChemRateLimitError:
                raise
            except Exception:
                time.sleep(5)

        # Batch failed after retries: try each CID individually so one bad ID
        # does not drop the rest of the chunk.
        if not ok:
            for cid in chunk:
                smiles = smiles_from_cid(cid)
                if smiles:
                    mapping[str(cid)] = smiles
                time.sleep(0.5)

        time.sleep(delay)
        if checkpoint is not None:
            pd.DataFrame(mapping.items(), columns=["CID", "SMILES"]).to_csv(
                checkpoint, index=False
            )
    return mapping


def pug_view(cid):
    """Full PubChem compound record (sections: safety, pharmacology, ...).

    This is the nested JSON walked by ``score_pug``. Returns None if the
    record is missing or the request fails.
    """
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def assay_summary(cid):
    """Bioassay summary JSON for a CID, or None on failure.

    Used by ``score_assay`` as a second evidence channel besides PUG-View.
    """
    try:
        r = _get(f"{PUBCHEM}/cid/{cid}/assaysummary/JSON")
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _count(text, keywords):
    """How many times each keyword appears in already-lowercased ``text``."""
    return sum(text.count(k.lower()) for k in keywords)


def score_pug(data):
    """Count pulmonary-ADR keyword hits in adverse-related PUG-View sections.

    Walks the nested ``Section`` tree. A section contributes only if its
    ``TOCHeading`` looks like an adverse/toxicity heading; indication or
    mechanism text is ignored. The integer is not a probability — phase 2
    only cares whether the total is zero.
    """
    if not data:
        return 0

    def walk(section):
        count = 0
        if any(term in section.get("TOCHeading", "").lower() for term in ADVERSE_TERMS):
            for info in section.get("Information", []):
                count += _count(json.dumps(info).lower(), KEYWORDS)
        for sub in section.get("Section", []):
            count += walk(sub)
        return count

    # Records usually nest sections under "Record"; some responses put them
    # at the top level.
    sections = data.get("Record", {}).get("Section", []) or data.get("Section", [])
    return sum(walk(section) for section in sections)


def score_assay(data):
    """Count pulmonary-ADR keywords in assays that also mention adverse terms.

    Same idea as ``score_pug``: a fibrosis assay for a therapeutic indication
    should not mark the compound as a pulmonary-ADR positive. The assay JSON
    must mention both an adverse/toxicity term and a pulmonary keyword.
    """
    if not data:
        return 0
    assays = data.get("Assays", []) or (data if isinstance(data, list) else [])
    return sum(
        _count(json.dumps(assay).lower(), KEYWORDS)
        for assay in assays
        if any(term in json.dumps(assay).lower() for term in ADVERSE_TERMS)
    )


def resolve_cid(pubchem_id, drug_name):
    """Pick a CID for scoring: stored ID first, then name lookup.

    Returns
    -------
    (cid, source) : (int or None, str)
        ``source`` is ``"PubChem_ID"``, ``"Drug_Name"``,
        ``"Drug_Name (fallback)"``, or ``"none"`` so the scored-negatives CSV
        records how the CID was obtained.

    Name lookup tries a stripped variant first (punctuation removed) because
    salt/hydrate suffixes often prevent an exact PubChem name match, then the
    original string.
    """
    value = None if pd.isna(pubchem_id) else str(pubchem_id).strip()
    # Placeholders written during phase-1 CID search are not real IDs.
    if value and value not in {"", "NAN", "None", "NOT_FOUND", "TO_BE_FOUND"}:
        try:
            return int(float(value)), "PubChem_ID"
        except ValueError:
            pass

    name = None if pd.isna(drug_name) else str(drug_name).strip()
    if not name:
        return None, "none"

    filtered = re.sub(r"[^a-zA-Z0-9\s\-]", "", name).strip() or name
    cid = cid_from_name(filtered)
    if cid:
        return cid, "Drug_Name"
    cid = cid_from_name(name)
    if cid:
        return cid, "Drug_Name (fallback)"
    return None, "none"
