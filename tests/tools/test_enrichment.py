import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

import pertpy as pt


@pytest.fixture
def dummy_adata():
    n_obs = 10
    n_vars = 5
    rng = np.random.default_rng()
    X = rng.random((n_obs, n_vars))
    adata = AnnData(X)
    adata.var_names = [f"gene{i}" for i in range(n_vars)]
    adata.obs["cluster"] = ["group_1"] * 5 + ["group_2"] * 5
    sc.tl.rank_genes_groups(adata, groupby="cluster", method="t-test")

    return adata


@pytest.fixture(scope="module")
def enricher():
    return pt.tl.Enrichment()


def test_score_basic(dummy_adata, enricher):
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enricher.score(adata=dummy_adata, targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_different_layers(dummy_adata, enricher):
    rng = np.random.default_rng()
    dummy_adata.layers["layer"] = rng.random((10, 5))
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enricher.score(adata=dummy_adata, layer="layer", targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_nested_targets(dummy_adata, enricher):
    targets = {"category1": {"group1": ["gene1", "gene2"]}, "category2": {"group2": ["gene3", "gene4"]}}
    enricher.score(adata=dummy_adata, targets=targets, nested=True)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_hypergeometric_basic(dummy_adata, enricher):
    targets = {"group1": ["gene1", "gene2"]}
    results = enricher.hypergeometric(dummy_adata, targets=targets)
    assert isinstance(results, dict)


def test_hypergeometric_with_nested_targets(dummy_adata, enricher):
    targets = {"category1": {"group1": ["gene1", "gene2"]}}
    results = enricher.hypergeometric(dummy_adata, targets=targets, nested=True)
    assert isinstance(results, dict)


@pytest.mark.parametrize("direction", ["up", "down", "both"])
def test_hypergeometric_with_different_directions(dummy_adata, enricher, direction):
    targets = {"group1": ["gene1", "gene2"]}
    results = enricher.hypergeometric(dummy_adata, targets=targets, direction=direction)
    assert isinstance(results, dict)


def test_signature_reversal_cmap_writes_scores_to_adata(enricher):
    labels = ["reverse", "mimic", "control"]
    adata = AnnData(
        X=np.array([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]),
        obs=pd.DataFrame({"perturbation": labels}, index=labels),
    )
    adata.var_names = ["disease_up", "disease_down"]

    enricher.signature_reversal(
        adata,
        up_genes=["disease_up"],
        down_genes=["disease_down"],
    )

    assert adata.obs["signature_reversal_rank"].idxmin() == "reverse"
    assert adata.obs.loc["reverse", "signature_reversal_score"] > adata.obs.loc["mimic", "signature_reversal_score"]
    assert adata.obs.loc["reverse", "signature_reversal_connectivity"] < 0
    np.testing.assert_allclose(adata.obs.loc["control", "signature_reversal_connectivity"], 0.0)
    assert adata.uns["signature_reversal"]["matched_genes"] == ["disease_up", "disease_down"]


def test_signature_reversal_accepts_signed_query(enricher):
    adata = AnnData(
        X=np.array([[-1.0, 1.0], [1.0, -1.0]]),
        obs=pd.DataFrame(index=["reverse", "mimic"]),
    )
    adata.var_names = ["up", "down"]

    enricher.signature_reversal(
        adata,
        pd.Series({"up": 1.0, "down": -1.0}),
    )

    assert adata.obs["signature_reversal_rank"].idxmin() == "reverse"
    np.testing.assert_allclose(adata.obs.loc["reverse", "signature_reversal_score"], 1.0)
    np.testing.assert_allclose(adata.obs.loc["mimic", "signature_reversal_score"], -1.0)
