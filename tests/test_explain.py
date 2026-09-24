"""Step 9 of the modelling guide: the faithfulness test accepts explanations that cite the payload and rejects invented facts."""

from ranking.llm.explain import faithfulness, make_payload, templated_explanation

PAYLOAD = make_payload(2, 12345, "Dresses", 0.412, 0.081, 0.017, 0.093, {"co-visitation": 0.31, "popularity": 0.12, "recency": -0.02, "cart rate": 0.01})


def test_the_templated_explanation_is_faithful_by_construction():
    ok, nums, groups = faithfulness(templated_explanation(PAYLOAD), PAYLOAD)
    assert ok and not nums and not groups


def test_percentages_of_payload_probabilities_are_supported():
    assert faithfulness("Ranked 2 in Dresses; click chance 41.2% driven by co-visitation.", PAYLOAD)[0]


def test_an_invented_number_is_rejected():
    ok, nums, _ = faithfulness("Ranked 2 with a click probability of 0.77, driven by co-visitation.", PAYLOAD)
    assert not ok and "0.77" in nums


def test_an_invented_feature_group_is_rejected():
    ok, _, groups = faithfulness("Ranked 2, mostly because of two-tower similarity and co-visitation.", PAYLOAD)
    assert not ok and "two-tower similarity" in groups


def test_a_group_outside_the_top_three_is_not_supported():
    ok, _, groups = faithfulness("Ranked 2; the cart rate matters most.", PAYLOAD)
    assert not ok and "cart rate" in groups
