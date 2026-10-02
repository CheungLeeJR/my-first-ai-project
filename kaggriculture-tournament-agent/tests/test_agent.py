import agent


def test_empty_observation_falls_back_to_legal_pass():
    assert agent.agent({}) == {"farmer": ["PASS"], "hands": [], "market": []}


def test_opening_layout_matches_declared_mix():
    melon, wheat, carrot = agent._opening_maps(10)
    assert len(melon) == agent.OPENING_MELONS == 9
    assert len(wheat) == agent.OPENING_WHEAT == 10
    assert len(carrot) == agent.OPENING_CARROTS == 2
    assert not (melon & wheat or melon & carrot or wheat & carrot)


def test_market_price_decreases_as_public_inventory_rises():
    market = {}
    for item, params in agent.BASE_MARKET_PARAMS.items():
        i0 = params["I0"]
        assert agent._market_price(item, i0 - 100, market) >= agent._market_price(item, i0, market)
        assert agent._market_price(item, i0, market) >= agent._market_price(item, i0 + 100, market)


def test_long_term_roles_are_disjoint_and_exclude_animal_sites():
    strawberry, melon, wheat, flex = agent._long_term_site_sets(10)
    role_sets = [strawberry, melon, wheat, flex]
    for i, left in enumerate(role_sets):
        assert not (left & set(agent.ANIMAL_SITES))
        for right in role_sets[i + 1 :]:
            assert not (left & right)


def test_final_agent_fallback_matches_visible_hand_count():
    obs = {"player": 0, "farms": [{"hands": [[0, 0], [1, 1]], "tiles": []}], "private": {}}
    result = agent.agent(obs)
    assert set(result) == {"farmer", "hands", "market"}
    assert len(result["hands"]) == 2


def test_safe_sell_never_exceeds_available_stock():
    for item in agent.SALEABLE:
        qty = agent._safe_sell_qty(item, stock=7, start_inventory=10000, market={}, day=10)
        assert 0 <= qty <= 7


def test_same_observation_is_deterministic():
    obs = {"player": 0, "farms": [{"hands": [], "tiles": []}], "private": {}}
    assert agent.agent(obs) == agent.agent(obs)
