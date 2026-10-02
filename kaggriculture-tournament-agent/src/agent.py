"""Kaggriculture high-throughput tournament agent (V3.3).

Standalone, standard-library-only submission.  The policy is built for the
30-day / 720-step Advanced environment and emphasizes:

* a proven dense opening (9 melon, 10 wheat, 2 carrot);
* staged 3-quadrant expansion and 13 daily hands when the farm can fund it;
* 14 near-shed pasture sites with a diversified cow/sheep portfolio;
* 34 strawberry-capable field sites plus feed wheat and demand flex tiles;
* feed/care/fertilizer/harvest logistics with emergency survival priorities;
* exact public market-curve pricing and demand-timed premium liquidation;
* opponent-visible production pressure in animal mix and sell timing;
* terminal-day harvest/return/liquidation safeguards.

Required Kaggle entry point: ``agent(obs)``.
"""

from math import inf, sqrt, log1p, log10
from functools import lru_cache


# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

TOTAL_DAYS = 30
TURNS_PER_DAY = 24
FINAL_ACTION_STEP = 718
MAX_ORDERS = 10
SHED_CAPACITY = 100

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "max_day": 4,  "max_yield": 6, "ongoing": False, "interval": 0},
    "CARROT":     {"seed": 20,  "first": 2,  "max_day": 3,  "max_yield": 4, "ongoing": False, "interval": 0},
    "TOMATO":     {"seed": 50,  "first": 8,  "max_day": 8,  "max_yield": 4, "ongoing": True,  "interval": 1},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "max_yield": 4, "ongoing": True,  "interval": 2},
    "MELON":      {"seed": 80,  "first": 10, "max_day": 12, "max_yield": 6, "ongoing": False, "interval": 0},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "product": "EGG",  "structure": "COOP",    "first": 4, "interval": 1, "max_held": 4},
    "COW":   {"cost": 400, "product": "MILK", "structure": "PASTURE", "first": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"cost": 500, "product": "WOOL", "structure": "PASTURE", "first": 6, "interval": 3, "max_held": 6},
}

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
PREMIUM = ("MILK", "STRAWBERRY", "WOOL", "MELON")
SALEABLE = ("MILK", "WOOL", "STRAWBERRY", "MELON", "CARROT", "TOMATO", "EGG", "FERTILIZER", "WHEAT")

BASE_MARKET_PARAMS = {
    "WHEAT":      {"base": 25,  "I0": 10000, "T": 400, "below_func": "sqrt",  "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base": 35,  "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base": 60,  "I0": 10000, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt",  "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base": 50,  "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt",  "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear","below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

LAND_PRICES = (1000, 2000, 4000)
HIRE_COSTS = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597)

# Four NW, five NE, five SW.  These are deliberately near the central shed.
ANIMAL_SITES = (
    (4, 2), (4, 3), (3, 4), (4, 4),
    (6, 2), (5, 3), (7, 3), (5, 4), (7, 4),
    (3, 5), (4, 5), (3, 6), (4, 6), (4, 7),
)

OPENING_MELONS = 9
OPENING_WHEAT = 10
OPENING_CARROTS = 2
STRAWBERRY_TARGET = 38
ROTATION_MELON_TILES = 8
FEED_WHEAT_TILES = 7
FINAL_ANIMAL_TARGET = 14


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dict(obj):
    return obj if isinstance(obj, dict) else {}


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _dist(a, b):
    return abs(_int(a[0]) - _int(b[0])) + abs(_int(a[1]) - _int(b[1]))


def _quadrant(pos, board_size=10):
    x, y = pos
    h = board_size // 2
    return ("N" if y < h else "S") + ("W" if x < h else "E")


@lru_cache(maxsize=None)
def _shed_tiles(board_size=10):
    h = board_size // 2
    return ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


def _nearest_shed(pos, board_size=10):
    return min(_shed_tiles(board_size), key=lambda p: (_dist(pos, p), p[1], p[0]))


@lru_cache(maxsize=None)
def _shed_distance(pos, board_size=10):
    return _dist(pos, _nearest_shed(pos, board_size))


def _move_towards(pos, target):
    x, y = _int(pos[0]), _int(pos[1])
    tx, ty = _int(target[0]), _int(target[1])
    # Choose the larger remaining axis first to reduce long-axis oscillation.
    dx, dy = tx - x, ty - y
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _step_day_hour(obs):
    step = _int(_get(obs, "step", -1), -1)
    if step < 0:
        day = _int(_get(obs, "day", 0))
        hour = _int(_get(obs, "hour", 0))
        step = day * TURNS_PER_DAY + hour
    else:
        day = _int(_get(obs, "day", step // TURNS_PER_DAY), step // TURNS_PER_DAY)
        hour = _int(_get(obs, "hour", step % TURNS_PER_DAY), step % TURNS_PER_DAY)
    return step, day, hour


def _shape(name, x, T):
    x = max(0.0, float(x))
    if name == "linear":
        return x
    if name == "sq":
        return x * x
    if name == "sqrt":
        return sqrt(x)
    if name == "log":
        return log1p(x)
    if name == "log10":
        return log10(1.0 + x)
    if name == "hinge":
        if not T or T <= 0:
            return x
        u = x / float(T)
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _market_params(market, item):
    p = dict(BASE_MARKET_PARAMS[item])
    overrides = _dict(_get(market, "params", {}))
    patch = overrides.get(item)
    if isinstance(patch, dict):
        p.update(patch)
    return p


def _market_price(item, inventory, market):
    p = _market_params(market, item)
    base = float(p["base"])
    i0 = _int(p["I0"], 10000)
    T = float(p["T"])
    if inventory < i0:
        f = p["below_func"]
        denom = _shape(f, T, T) or 1.0
        amp = float(p["below_target"]) * base / denom
        price = base + amp * _shape(f, i0 - inventory, T)
    else:
        f = p["above_func"]
        denom = _shape(f, T, T) or 1.0
        amp = float(p["above_target"]) * base / denom
        price = base - amp * _shape(f, inventory - i0, T)
    return max(1, int(round(price)))


def _town_tick_demand(obs, item):
    town = _dict(_get(obs, "town", {}))
    demand = 0
    for shop in list(town.get("unlocked_shops", []) or []):
        products = SHOPS.get(shop, ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _unlocked_positions(farm):
    out = []
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row):
            if tile != "LOCKED":
                out.append((x, y))
    return out


@lru_cache(maxsize=None)
def _all_crop_sites(board_size=10):
    aset = set(ANIMAL_SITES)
    sites = []
    for y in range(board_size):
        for x in range(board_size):
            p = (x, y)
            if p in aset or _quadrant(p, board_size) == "SE":
                continue
            # Three-quadrant route only.
            sites.append(p)
    return sites


@lru_cache(maxsize=None)
def _opening_maps(board_size=10):
    nw = [p for p in _all_crop_sites(board_size) if _quadrant(p, board_size) == "NW"]
    # Outer/mid-distance tiles carry the slow melon wave; inner tiles cycle faster.
    ranked_outer = sorted(nw, key=lambda p: (-_shed_distance(p, board_size), p[1], p[0]))
    melon = set(ranked_outer[:OPENING_MELONS])
    remain = [p for p in sorted(nw, key=lambda p: (_shed_distance(p, board_size), p[1], p[0])) if p not in melon]
    carrot = set(remain[:OPENING_CARROTS])
    wheat = set(p for p in nw if p not in melon and p not in carrot)
    return melon, wheat, carrot


@lru_cache(maxsize=None)
def _long_term_site_sets(board_size=10):
    sites = _all_crop_sites(board_size)
    # Inner sites are ideal for strawberries (many harvest trips); the farthest
    # productive sites remain melons because melon value per worker-action is high
    # despite the long watering cycle.  A compact wheat belt supplies feed.
    by_close = sorted(sites, key=lambda p: (_shed_distance(p, board_size), _quadrant(p, board_size), p[1], p[0]))
    strawberries = set(by_close[:STRAWBERRY_TARGET])
    remainder = [p for p in sites if p not in strawberries]
    by_far = sorted(remainder, key=lambda p: (-_shed_distance(p, board_size), p[1], p[0]))
    melons = set(by_far[:ROTATION_MELON_TILES])
    remainder = [p for p in remainder if p not in melons]
    by_feed = sorted(remainder, key=lambda p: (_shed_distance(p, board_size), p[1], p[0]))
    wheat = set(by_feed[:FEED_WHEAT_TILES])
    flex = set(p for p in sites if p not in strawberries and p not in melons and p not in wheat)
    return strawberries, melons, wheat, flex


# ---------------------------------------------------------------------------
# Farm / opponent state
# ---------------------------------------------------------------------------

def _scan_farm(farm):
    plants = {c: 0 for c in CROPS}
    animals = {a: 0 for a in ANIMALS}
    ready = {p: 0 for p in PRODUCTS}
    weeds = 0
    empty_pastures = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                weeds += 1
            if kind == "PLANT":
                crop = tile.get("crop")
                if crop in plants:
                    plants[crop] += 1
                    ready[crop] += max(0, _int(tile.get("yield_units", 0)))
            animal = tile.get("animal")
            if animal in animals:
                animals[animal] += 1
                prod = ANIMALS[animal]["product"]
                ready[prod] += max(0, _int(tile.get("yield_units", 0)))
            elif kind == "PASTURE":
                empty_pastures += 1
    return {
        "plants": plants,
        "animals": animals,
        "ready": ready,
        "weeds": weeds,
        "empty_pastures": empty_pastures,
    }


def _opponent_state(obs, player):
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return {"money": 0.0, "plants": {c: 0 for c in CROPS}, "animals": {a: 0 for a in ANIMALS}, "ready": {p: 0 for p in PRODUCTS}}
    opp = 1 - player
    if not (0 <= opp < len(farms)):
        return {"money": 0.0, "plants": {c: 0 for c in CROPS}, "animals": {a: 0 for a in ANIMALS}, "ready": {p: 0 for p in PRODUCTS}}
    scan = _scan_farm(_dict(farms[opp]))
    scan["money"] = float(_dict(farms[opp]).get("money", 0) or 0)
    return scan


def _count_all_owned_animal(private, farm, animal):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") == animal:
                n += 1
    n += _int(_dict(private.get("shed", {})).get(animal, 0))
    for inv in list(private.get("inventories", []) or []):
        n += _int(_dict(inv).get(animal, 0))
    return n


def _placed_animal_count(farm):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                n += 1
    return n


def _unfed_animals(farm):
    out = []
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("animal") in ANIMALS and not tile.get("fed_today", False):
                out.append(((x, y), tile))
    return out


def _target_total_animals(day, unlocked):
    max_sites = 4
    if "NE" in unlocked:
        max_sites = 9
    if "SW" in unlocked:
        max_sites = 14
    # Capital goes into productive livestock before expensive day-labour.
    if day < 6:
        target = 4
    elif day < 8:
        target = 6
    elif day < 11:
        target = 9 if "SW" not in unlocked else 12
    elif day < 23:
        target = FINAL_ANIMAL_TARGET
    else:
        target = 0
    return min(max_sites, target)


def _desired_species(obs, farm, private, opponent, day, unlocked):
    current_cows = _count_all_owned_animal(private, farm, "COW")
    current_sheep = _count_all_owned_animal(private, farm, "SHEEP")
    current_total = current_cows + current_sheep
    total_target = _target_total_animals(day, unlocked)
    if day >= 23:
        return current_cows, current_sheep
    shops = list(_dict(_get(obs, "town", {})).get("unlocked_shops", []) or [])
    # Public top-route analysis finds shop order to be a more stable causal signal
    # than current price (which is itself affected by players' chosen portfolios).
    # Keep the validated balanced 8/6 footprint unless YARN appears immediately.
    if "YARN_STORE" in shops[:2]:
        final_cows, final_sheep = 6, 8
    else:
        final_cows, final_sheep = 8, 6
    market = _dict(_get(obs, "market", {}))
    prices = _dict(market.get("prices", {}))
    milk_rel = float(prices.get("MILK", 160) or 160) / 160.0
    wool_rel = float(prices.get("WOOL", 200) or 200) / 200.0

    # Scale the final mix to the current expansion phase, never asking to remove stock.
    cow_target = int(round(total_target * final_cows / float(final_cows + final_sheep)))
    cow_target = max(current_cows, min(total_target, cow_target))
    sheep_target = max(current_sheep, total_target - cow_target)
    if cow_target + sheep_target < total_target:
        if milk_rel >= wool_rel:
            cow_target += total_target - cow_target - sheep_target
        else:
            sheep_target += total_target - cow_target - sheep_target
    # If current holdings already overshoot one side, fill the other side only.
    while cow_target + sheep_target > max(total_target, current_total):
        if cow_target > current_cows and cow_target - current_cows >= sheep_target - current_sheep:
            cow_target -= 1
        elif sheep_target > current_sheep:
            sheep_target -= 1
        else:
            break
    return cow_target, sheep_target


# ---------------------------------------------------------------------------
# Crop plan and work generation
# ---------------------------------------------------------------------------

def _flex_crop(obs, opponent, farm_scan):
    _, day, _ = _step_day_hour(obs)
    market = _dict(_get(obs, "market", {}))
    prices = _dict(market.get("prices", {}))
    shops = list(_dict(_get(obs, "town", {})).get("unlocked_shops", []) or [])
    pet = shops.count("PET_CAFE")
    farmers = shops.count("FARMERS_MARKET")
    pizza = shops.count("PIZZA_SHOP")
    bakery = shops.count("BAKERY")
    brunch = shops.count("BRUNCH_SPOT")
    ice = shops.count("ICE_CREAM_SHOP")

    # Carrot is excellent once pet-cafe / farmers-market demand pushes the hinge.
    carrot_signal = 2 * pet + farmers + max(0.0, (float(prices.get("CARROT", 35) or 35) / 35.0 - 1.0) * 2.0)
    # Tomato only enters when repeated town demand exists and the rival is not already flooding it.
    tomato_signal = pizza + farmers + max(0.0, (float(prices.get("TOMATO", 60) or 60) / 60.0 - 1.0) * 1.5)
    tomato_signal -= 0.18 * _int(opponent["plants"].get("TOMATO", 0))
    wheat_signal = 0.55 * (bakery + brunch + ice + pizza + farmers)
    wheat_signal += max(0.0, (float(prices.get("WHEAT", 25) or 25) / 25.0 - 1.0))
    wheat_signal += 0.12 * max(0, _placed_animal_count_from_scan(farm_scan) - FEED_WHEAT_TILES)

    if day <= 17 and tomato_signal >= max(2.4, carrot_signal + 0.5, wheat_signal + 0.8):
        return "TOMATO"
    if carrot_signal >= max(1.8, wheat_signal + 0.25):
        return "CARROT"
    return "WHEAT"


def _placed_animal_count_from_scan(scan):
    return sum(_int(v) for v in scan.get("animals", {}).values())


def _crop_role(pos, obs, farm, opponent, farm_scan, board_size=10):
    _, day, _ = _step_day_hour(obs)
    q = _quadrant(pos, board_size)
    if q not in set(farm.get("unlocked_quadrants", []) or ["NW"]):
        return None
    if pos in set(ANIMAL_SITES):
        return None

    opening_melon, opening_wheat, opening_carrot = _opening_maps(board_size)
    if day <= 3 and q == "NW":
        if pos in opening_melon:
            return "MELON"
        if pos in opening_carrot:
            return "CARROT"
        if pos in opening_wheat:
            return "WHEAT"

    strawberry_sites, melon_sites, wheat_sites, flex_sites = _long_term_site_sets(board_size)
    days_left = TOTAL_DAYS - day

    if pos in strawberry_sites and day <= 18 and days_left >= 11:
        return "STRAWBERRY"
    if pos in melon_sites and day <= 16 and days_left >= 11:
        return "MELON"
    if pos in wheat_sites and day <= 25 and days_left >= 4:
        return "WHEAT"
    if pos in flex_sites:
        role = _flex_crop(obs, opponent, farm_scan)
        if role == "TOMATO" and day > 17:
            role = "CARROT" if days_left >= 3 else None
        if role == "CARROT" and days_left < 4:
            return None
        if role == "WHEAT" and days_left < 5:
            return None
        return role

    # Strawberry slots switch to short-cycle crops once a late replant would not pay.
    if pos in strawberry_sites and day > 18:
        if days_left >= 4:
            return "CARROT" if _town_tick_demand(obs, "CARROT") >= 2 else "WHEAT"
    return None


def _one_time_ready(tile, day, final_day=False):
    crop = tile.get("crop")
    if crop not in CROPS or CROPS[crop]["ongoing"]:
        return False
    units = _int(tile.get("yield_units", 0))
    if units <= 0:
        return False
    age = day - _int(tile.get("planted_day", day))
    if final_day:
        return age >= CROPS[crop]["first"]
    if crop == "MELON":
        return age >= 10
    return age >= CROPS[crop]["max_day"]


def _strawberry_fertilize_due(tile, day):
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT" or tile.get("crop") != "STRAWBERRY":
        return False
    age = day - _int(tile.get("planted_day", day))
    if age < 0 or age > 15:
        return False
    # Production occurs at the end of ages 9, 11, 13, 15.  A 3-day fertilizer
    # application at age 9 covers 9+11; another at 13 covers 13+15.
    if age not in (9, 13):
        return False
    return _int(tile.get("fertilized_until_day", -1), -1) < day + 2


def _tomato_fertilize_due(tile, day, obs):
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT" or tile.get("crop") != "TOMATO":
        return False
    age = day - _int(tile.get("planted_day", day))
    if age != 7:
        return False
    market = _dict(_get(obs, "market", {}))
    p = _int(_dict(market.get("prices", {})).get("TOMATO", 60), 60)
    fert = _int(_dict(market.get("prices", {})).get("FERTILIZER", 100), 100)
    return 3 * p >= int(1.35 * fert)


def _build_tasks(obs, farm, private, opponent):
    step, day, hour = _step_day_hour(obs)
    board_size = len(farm.get("tiles", []) or []) or 10
    final_day = day >= TOTAL_DAYS - 1
    farm_scan = _scan_farm(farm)
    tasks = []
    unlocked = set(farm.get("unlocked_quadrants", []) or ["NW"])
    active_animal_sites = [p for p in ANIMAL_SITES if _quadrant(p, board_size) in unlocked]
    animal_set = set(active_animal_sites)
    tiles = farm.get("tiles", []) or []
    seeds = _dict(private.get("seeds", {}))

    # Animal sites and animal care.
    for pos in active_animal_sites:
        x, y = pos
        tile = tiles[y][x]
        q = _quadrant(pos, board_size)
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            tasks.append((5200, pos, ["DIG"], None, "ANIMAL", q))
            continue
        if tile is None:
            tasks.append((3300, pos, ["BUILD_PASTURE"], None, "ANIMAL", q))
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PASTURE" and not tile.get("animal"):
            # Placement is handled as forced cargo logistics in _assign_units.
            continue
        animal = tile.get("animal") if isinstance(tile, dict) else None
        if animal not in ANIMALS:
            continue

        if final_day:
            if _int(tile.get("yield_units", 0)) > 0:
                tasks.append((6100, pos, ["HARVEST"], None, "ANIMAL", q))
            if tile.get("fertilizer_available", False):
                tasks.append((4700, pos, ["COLLECT_FERTILIZER"], None, "ANIMAL", q))
            continue

        if not tile.get("fed_today", False):
            missed = _int(tile.get("consecutive_unfed", 0))
            tasks.append((6000 + 900 * min(1, missed) + 18 * hour, pos, ["FEED"], ("INV", "WHEAT"), "ANIMAL", q))
        yld = _int(tile.get("yield_units", 0))
        if yld > 0:
            # Care bonuses often fill cow/sheep tiles to the cap; never let the next tick overwrite value.
            pr = 5600 if yld >= 4 else 4300
            tasks.append((pr + 8 * hour, pos, ["HARVEST"], None, "ANIMAL", q))
        if tile.get("fertilizer_available", False):
            tasks.append((3550 + 10 * hour, pos, ["COLLECT_FERTILIZER"], None, "ANIMAL", q))
        if not tile.get("cared_today", False):
            tasks.append((3900 + 15 * hour, pos, ["CARE"], None, "ANIMAL", q))

    # Crops and weeds.
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile == "LOCKED":
                continue
            pos = (x, y)
            if pos in animal_set:
                continue
            q = _quadrant(pos, board_size)
            role = _crop_role(pos, obs, farm, opponent, farm_scan, board_size)

            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                if role is not None and not final_day:
                    tasks.append((3150, pos, ["DIG"], None, "CROP", q))
                continue

            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                yld = _int(tile.get("yield_units", 0))
                if final_day:
                    if yld > 0 and (CROPS.get(crop, {}).get("ongoing") or day - _int(tile.get("planted_day", day)) >= CROPS.get(crop, {}).get("first", 999)):
                        tasks.append((5900, pos, ["HARVEST"], None, "CROP", q))
                    continue

                if _one_time_ready(tile, day, False):
                    pr = 5200 if crop == "MELON" else 4200
                    tasks.append((pr, pos, ["HARVEST"], None, "CROP", q))
                elif CROPS.get(crop, {}).get("ongoing") and yld > 0:
                    age = day - _int(tile.get("planted_day", day))
                    pr = 4700 if yld >= 3 or age >= 15 or hour >= 16 else 3600
                    tasks.append((pr, pos, ["HARVEST"], None, "CROP", q))

                # Fertilizer must be done before the end-of-day production tick.
                if _strawberry_fertilize_due(tile, day) or _tomato_fertilize_due(tile, day, obs):
                    tasks.append((4500 + 10 * hour, pos, ["FERTILIZE"], ("INV", "FERTILIZER"), "CROP", q))

                if not tile.get("watered_today", False):
                    planted_today = _int(tile.get("planted_day", day)) == day
                    missed = _int(tile.get("consecutive_unwatered", 0)) >= 1
                    pr = 5750 if planted_today else (5450 if missed else 3700 + 18 * hour)
                    tasks.append((pr, pos, ["WATER"], None, "CROP", q))
                continue

            if tile is None and role is not None and not final_day and hour <= 18:
                # Do not plant beyond a realistic cash-return window.
                days_left = TOTAL_DAYS - day
                if role == "MELON" and day > 16:
                    continue
                if role == "STRAWBERRY" and day > 18:
                    continue
                if role == "TOMATO" and day > 17:
                    continue
                maturity = CROPS[role]["first"]
                if days_left <= maturity:
                    continue
                if _int(seeds.get(role, 0)) > 0:
                    pr = 4200 if role == "MELON" else (3500 if role == "STRAWBERRY" else 2800)
                    tasks.append((pr, pos, ["PLANT", role], ("SEED", role), "CROP", q))

    return tasks


# ---------------------------------------------------------------------------
# Unit logistics / task assignment
# ---------------------------------------------------------------------------

def _inventory_sale_load(inv):
    inv = _dict(inv)
    return sum(_int(inv.get(item, 0)) for item in PRODUCTS if item != "WHEAT")


def _inventory_premium_load(inv):
    inv = _dict(inv)
    return sum(_int(inv.get(item, 0)) for item in PREMIUM)


def _empty_pastures(farm):
    out = []
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PASTURE" and not tile.get("animal"):
                out.append((x, y))
    return out


def _unit_preferred_quadrant(index, unlocked):
    unlocked = [q for q in ("NW", "NE", "SW") if q in set(unlocked)]
    if not unlocked:
        return "NW"
    # Round-robin spreads workers evenly over the three equally-sized work zones.
    return unlocked[index % len(unlocked)]


def _task_capable(task, inv, seed_left):
    req = task[3]
    if req is None:
        return True
    kind, item = req
    if kind == "INV":
        return _int(_dict(inv).get(item, 0)) > 0
    if kind == "SEED":
        return _int(seed_left.get(item, 0)) > 0
    return True


def _project_drop(projected, inv):
    room = max(0, SHED_CAPACITY - sum(max(0, _int(v)) for v in projected.values()))
    if room <= 0:
        return
    # Mirrors DROP insertion order well enough for sale projection; inventories are tiny.
    for item, raw in _dict(inv).items():
        n = max(0, _int(raw))
        if n <= 0 or room <= 0:
            continue
        take = min(n, room)
        projected[item] = _int(projected.get(item, 0)) + take
        room -= take


def _assign_units(obs, farm, private, tasks):
    step, day, hour = _step_day_hour(obs)
    final_day = day >= TOTAL_DAYS - 1
    board_size = len(farm.get("tiles", []) or []) or 10
    shed_tiles = set(_shed_tiles(board_size))
    shed = _dict(private.get("shed", {}))
    projected_shed = {k: _int(v) for k, v in shed.items()}
    inventories = list(private.get("inventories", []) or [])
    positions = [farm.get("farmer", [board_size // 2 - 1, board_size // 2 - 1])] + list(farm.get("hands", []) or [])
    while len(inventories) < len(positions):
        inventories.append({})

    unlocked = list(farm.get("unlocked_quadrants", []) or ["NW"])
    empty_pastures = _empty_pastures(farm)
    claimed = set()
    actions = []
    seed_left = {k: _int(v) for k, v in _dict(private.get("seeds", {})).items()}

    unfed = _unfed_animals(farm) if not final_day else []
    carried_wheat = sum(_int(_dict(inv).get("WHEAT", 0)) for inv in inventories)
    feed_shortfall = max(0, len(unfed) - carried_wheat)
    emergency_unfed = sum(1 for _, tile in unfed if _int(tile.get("consecutive_unfed", 0)) >= 1)
    feed_fetchers_needed = (feed_shortfall + 7) // 8
    if emergency_unfed:
        feed_fetchers_needed = max(feed_fetchers_needed, min(3, (emergency_unfed + 3) // 4))
    if hour >= 12 and feed_shortfall:
        feed_fetchers_needed = max(feed_fetchers_needed, min(3, (len(unfed) + 5) // 6))

    # Preselect nearest units without critical cargo to replenish feed.
    fetch_candidates = []
    for idx, pos0 in enumerate(positions):
        inv = _dict(inventories[idx])
        if _int(inv.get("WHEAT", 0)) > 0 or any(_int(inv.get(a, 0)) > 0 for a in ("COW", "SHEEP")):
            continue
        pos = (_int(pos0[0]), _int(pos0[1]))
        fetch_candidates.append((_shed_distance(pos, board_size), idx))
    feed_fetchers = {idx for _, idx in sorted(fetch_candidates)[:feed_fetchers_needed]}

    fertilizer_tasks = sum(1 for t in tasks if t[2] and t[2][0] == "FERTILIZE")
    carried_fert = sum(_int(_dict(inv).get("FERTILIZER", 0)) for inv in inventories)
    fert_fetcher = None
    if fertilizer_tasks > carried_fert and _int(shed.get("FERTILIZER", 0)) > 0 and not final_day:
        choices = []
        for idx, pos0 in enumerate(positions):
            inv = _dict(inventories[idx])
            if _inventory_sale_load(inv) > 0 or any(_int(inv.get(a, 0)) > 0 for a in ("COW", "SHEEP")):
                continue
            choices.append((_shed_distance((_int(pos0[0]), _int(pos0[1])), board_size), idx))
        if choices:
            fert_fetcher = min(choices)[1]

    total_carried_animals = {
        a: sum(_int(_dict(inv).get(a, 0)) for inv in inventories) for a in ("COW", "SHEEP")
    }

    for idx, pos0 in enumerate(positions):
        pos = (_int(pos0[0]), _int(pos0[1]))
        inv = _dict(inventories[idx])
        at_shed = pos in shed_tiles

        # 1) Animal cargo is expensive and has exactly one useful destination.
        carried_animal = None
        if _int(inv.get("COW", 0)) > 0:
            carried_animal = "COW"
        elif _int(inv.get("SHEEP", 0)) > 0:
            carried_animal = "SHEEP"
        available_pastures = [p for p in empty_pastures if p not in claimed]
        if carried_animal and available_pastures:
            target = min(available_pastures, key=lambda p: (_dist(pos, p), p[1], p[0]))
            claimed.add(target)
            if pos == target:
                actions.append(["PLACE", carried_animal])
                total_carried_animals[carried_animal] = max(0, total_carried_animals[carried_animal] - 1)
                empty_pastures.remove(target)
            else:
                actions.append(_move_towards(pos, target))
            continue

        sale_load = _inventory_sale_load(inv)
        premium_load = _inventory_premium_load(inv)

        # 2) Drop monetizable cargo immediately when already at the shed.
        if at_shed and sale_load > 0:
            actions.append(["DROP"])
            _project_drop(projected_shed, inv)
            continue

        # 3) Final day: inventory, not chores, is the bottleneck.  Return everything sellable.
        if final_day and sale_load > 0:
            target = _nearest_shed(pos, board_size)
            actions.append(_move_towards(pos, target) if pos != target else ["DROP"])
            continue

        # 4) Feed logistics.  Selected fetchers return for enough wheat to service a cluster.
        if not final_day and idx in feed_fetchers and _int(inv.get("WHEAT", 0)) == 0 and feed_shortfall > 0:
            if at_shed:
                available = _int(projected_shed.get("WHEAT", 0))
                if available > 0:
                    n = min(8, available, max(1, feed_shortfall))
                    actions.append(["PICKUP", "WHEAT", n])
                    projected_shed["WHEAT"] = max(0, available - n)
                    feed_shortfall = max(0, feed_shortfall - n)
                    continue
            else:
                target = _nearest_shed(pos, board_size)
                actions.append(_move_towards(pos, target))
                continue

        # 5) Empty pastures + purchased animals: dispatch one or two carriers at a time.
        if not final_day and available_pastures and at_shed:
            animal_pick = None
            for a in ("COW", "SHEEP"):
                if _int(projected_shed.get(a, 0)) > 0:
                    animal_pick = a
                    break
            if animal_pick:
                n = min(2, _int(projected_shed.get(animal_pick, 0)))
                actions.append(["PICKUP", animal_pick, n])
                projected_shed[animal_pick] = max(0, _int(projected_shed.get(animal_pick, 0)) - n)
                total_carried_animals[animal_pick] += n
                continue

        # 6) Fertilizer staging for high-ROI strawberry/tomato production days.
        if not final_day and idx == fert_fetcher and _int(inv.get("FERTILIZER", 0)) == 0:
            if at_shed and _int(projected_shed.get("FERTILIZER", 0)) > 0:
                n = min(4, _int(projected_shed.get("FERTILIZER", 0)), max(1, fertilizer_tasks - carried_fert))
                actions.append(["PICKUP", "FERTILIZER", n])
                projected_shed["FERTILIZER"] = max(0, _int(projected_shed.get("FERTILIZER", 0)) - n)
                continue
            elif not at_shed:
                actions.append(_move_towards(pos, _nearest_shed(pos, board_size)))
                continue

        # 7) Emergency feed lane.  Normally the greedy scheduler is more profitable,
        # but an animal that already missed yesterday must be fed today or it escapes.
        # Only those red-alert tiles pre-empt crop/harvest work.
        if not final_day and _int(inv.get("WHEAT", 0)) > 0:
            red_feed = []
            for task in tasks:
                if not (task[2] and task[2][0] == "FEED") or task[1] in claimed:
                    continue
                tx, ty = task[1]
                tt = farm.get("tiles", [])[ty][tx]
                if isinstance(tt, dict) and (_int(tt.get("consecutive_unfed", 0)) >= 1 or hour >= 12):
                    red_feed.append(task)
            if red_feed:
                preferred = _unit_preferred_quadrant(idx, unlocked)
                target_task = min(red_feed, key=lambda t: (0 if t[5] == preferred else 1, _dist(pos, t[1]), t[1][1], t[1][0]))
                target = target_task[1]
                claimed.add(target)
                actions.append(["FEED"] if pos == target else _move_towards(pos, target))
                continue

        # 8) Return valuable harvests before the shed overflows or before a demand tick.
        if sale_load > 0 and (premium_load > 0 or sale_load >= 8 or hour >= 18):
            target = _nearest_shed(pos, board_size)
            if pos != target:
                actions.append(_move_towards(pos, target))
                continue

        # 9) Greedy but zone-sticky work assignment.
        preferred = _unit_preferred_quadrant(idx, unlocked)
        best = None
        best_score = -inf
        for task in tasks:
            pr, target, action, req, kind, q = task
            if target in claimed:
                continue
            if not _task_capable(task, inv, seed_left):
                continue
            # On the final day, don't start a harvest trip that cannot plausibly return.
            if final_day and action and action[0] in ("HARVEST", "COLLECT_FERTILIZER"):
                remaining = max(0, 22 - hour)
                if _dist(pos, target) + _shed_distance(target, board_size) + 2 > remaining:
                    continue
            d = _dist(pos, target)
            score = float(pr) - 30.0 * d
            if q == preferred:
                score += 170.0
            if kind == "ANIMAL" and idx % 4 == 0:
                score += 110.0
            if pos == target:
                score += 260.0
            # Carrying the required commodity should create natural route stickiness.
            if req and req[0] == "INV" and _int(inv.get(req[1], 0)) > 0:
                score += 140.0
            if score > best_score:
                best_score = score
                best = task

        if best is not None:
            _, target, action, req, _, _ = best
            claimed.add(target)
            if pos == target:
                actions.append(list(action))
                if action[0] == "PLANT" and len(action) >= 2:
                    crop = action[1]
                    seed_left[crop] = max(0, _int(seed_left.get(crop, 0)) - 1)
                elif action[0] == "FEED":
                    carried_wheat = max(0, carried_wheat - 1)
            else:
                actions.append(_move_towards(pos, target))
            continue

        # 10) If the farm still has feed work but this unit is empty, drift toward the shed.
        if not final_day and unfed and _int(inv.get("WHEAT", 0)) == 0 and _int(projected_shed.get("WHEAT", 0)) > 0:
            target = _nearest_shed(pos, board_size)
            actions.append(_move_towards(pos, target) if pos != target else ["PASS"])
            continue

        actions.append(["PASS"])

    farmer = actions[0] if actions else ["PASS"]
    hands = actions[1:] if len(actions) > 1 else []
    return farmer, hands, projected_shed


# ---------------------------------------------------------------------------
# Market strategy
# ---------------------------------------------------------------------------

def _desired_hands(day, farm):
    # Replay-derived front-runner cadence: the farmer is an additional worker.
    # Avoid paying the very expensive 13th day-hand (233 coins) every day.
    money = float(farm.get("money", 0) or 0)
    if day <= 5:
        target = 4
    elif day == 6:
        target = 8
    elif day <= 8:
        target = 10
    else:
        target = 12
    if money < 120:
        target = min(target, 6)
    elif money < 240:
        target = min(target, 8)
    elif money < 420:
        target = min(target, 10)
    return target


def _hire_total_cost(start_hires, add):
    total = 0
    for n in range(start_hires, start_hires + max(0, add)):
        total += HIRE_COSTS[n] if n < len(HIRE_COSTS) else HIRE_COSTS[-1] * 2
    return total


def _safe_sell_qty(item, stock, start_inventory, market, day, force=False):
    stock = max(0, _int(stock))
    if stock <= 0:
        return 0
    if force:
        return stock
    p = _market_params(market, item)
    base = float(p["base"])
    if item == "MELON":
        ratio = 0.72 if day < 24 else 0.50
    elif item in ("MILK", "WOOL", "STRAWBERRY"):
        if day < 20:
            ratio = 0.86
        elif day < 25:
            ratio = 0.72
        elif day < 27:
            ratio = 0.55
        else:
            ratio = 0.30
    elif item == "FERTILIZER":
        ratio = 0.58 if day < 24 else 0.38
    elif item in ("CARROT", "TOMATO", "EGG"):
        ratio = 0.48
    else:
        ratio = 0.55

    min_price = max(1, int(round(base * ratio)))
    q = 0
    # Selling one unit quotes at current pre-sale inventory; subsequent units see +1.
    while q < stock:
        if _market_price(item, start_inventory + q, market) < min_price:
            break
        q += 1
    return q


def _sell_candidates(obs, farm, private, projected_shed, opponent):
    step, day, hour = _step_day_hour(obs)
    market = _dict(_get(obs, "market", {}))
    inv_market = _dict(market.get("inventory", {}))
    shed_total = sum(max(0, _int(v)) for v in projected_shed.values())
    force_final = step >= FINAL_ACTION_STEP or day >= TOTAL_DAYS - 1
    fresh_shop_tick = (step % 4 == 1)
    fresh_daily_tick = (step % 24 == 1)
    pressure = shed_total >= 82
    placed_animals = _placed_animal_count(farm)
    feed_reserve = 0 if force_final else max(10, placed_animals * 2)

    candidates = []
    for item in SALEABLE:
        stock = max(0, _int(projected_shed.get(item, 0)))
        if stock <= 0:
            continue
        if item == "WHEAT" and not force_final:
            stock = max(0, stock - feed_reserve)
            if stock <= 0:
                continue
        if item == "FERTILIZER" and not force_final:
            # Keep a small reserve for the high-ROI strawberry fertilization windows.
            stock = max(0, stock - 5)
            if stock <= 0:
                continue

        start = _int(inv_market.get(item, 10000), 10000)
        base = _market_params(market, item)["base"]
        current_price = _market_price(item, start, market)
        ready_opp = max(0, _int(opponent.get("ready", {}).get(item, 0)))

        timing_ok = True
        if item in ("MILK", "WOOL", "STRAWBERRY") and not force_final:
            timing_ok = fresh_shop_tick or pressure or start < _int(_market_params(market, item)["I0"]) - 3 or ready_opp >= 4
        elif item == "MELON" and not force_final:
            timing_ok = fresh_daily_tick or pressure or current_price >= int(0.98 * base)
        elif item == "FERTILIZER" and not force_final:
            timing_ok = pressure or current_price >= 78 or day >= 25

        if not timing_ok:
            continue
        qty = _safe_sell_qty(item, stock, start, market, day, force_final)
        if qty <= 0 and pressure:
            qty = min(stock, 4)
        if qty <= 0:
            continue

        # Do not unload all premium inventory on one non-terminal turn; leave room for
        # future town demand to repair the price curve.
        if item in ("MILK", "WOOL", "STRAWBERRY") and not force_final:
            cap = 12 if fresh_shop_tick else 6
            qty = min(qty, cap)
        elif item == "MELON" and not force_final:
            qty = min(qty, 18)
        elif item == "FERTILIZER" and not force_final:
            qty = min(qty, 10)

        marginal_value = sum(_market_price(item, start + k, market) for k in range(qty))
        # Collision bonus: sell our ready premium before a visible rival pipeline can
        # dump into the same shared curve.
        collision = ready_opp * float(base) * (0.18 if item in PREMIUM else 0.05)
        candidates.append((marginal_value + collision, current_price, item, qty))

    candidates.sort(reverse=True)
    return candidates


def _role_seed_needs(obs, farm, private, opponent):
    _, day, hour = _step_day_hour(obs)
    if day >= 29:
        return {}
    farm_scan = _scan_farm(farm)
    seeds = _dict(private.get("seeds", {}))
    desired = {c: 0 for c in CROPS}
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row):
            if tile == "LOCKED":
                continue
            if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
                role = _crop_role((x, y), obs, farm, opponent, farm_scan, len(farm.get("tiles", []) or []) or 10)
                if role in desired:
                    desired[role] += 1
    needs = {}
    for crop, n in desired.items():
        need = max(0, n - _int(seeds.get(crop, 0)))
        if need > 0:
            needs[crop] = need
    return needs


def _market_orders(obs, farm, private, opponent, projected_shed):
    step, day, hour = _step_day_hour(obs)
    final_day = day >= TOTAL_DAYS - 1
    money = float(farm.get("money", 0) or 0)
    budget = money
    market = _dict(_get(obs, "market", {}))
    prices = _dict(market.get("prices", {}))
    shed = _dict(private.get("shed", {}))
    inventories = list(private.get("inventories", []) or [])
    unlocked = list(farm.get("unlocked_quadrants", []) or ["NW"])
    orders = []

    # 1) Realize the best sale opportunities first; 3-4 slots usually leaves enough
    # space to finish hires and capital purchases across consecutive turns.
    sale_candidates = _sell_candidates(obs, farm, private, projected_shed, opponent)
    sale_cap = MAX_ORDERS if step >= FINAL_ACTION_STEP else (4 if day >= 4 else 3)
    for _, _, item, qty in sale_candidates[:sale_cap]:
        if qty > 0:
            orders.append(["SELL", item, qty])
    if step >= FINAL_ACTION_STEP:
        # Terminal turn: every remaining saleable item is worth more sold at $1 than held.
        already = {o[1]: _int(o[2]) for o in orders if len(o) >= 3 and o[0] == "SELL"}
        for item in SALEABLE:
            if len(orders) >= MAX_ORDERS:
                break
            stock = max(0, _int(projected_shed.get(item, 0)) - already.get(item, 0))
            if stock > 0:
                orders.append(["SELL", item, stock])
        return orders[:MAX_ORDERS]

    # Sales are not credited in our budget estimate: purchases therefore remain safe
    # even if an opponent moves the quote against us during lockstep processing.
    reserve = 180 if day <= 4 else 300

    # 2) Hires.  Cap hires per market turn so seeds/animals/feed are not starved by the
    # 10-order limit; the remainder is hired on the next one or two turns.
    desired_hands = _desired_hands(day, farm)
    current_hands = len(farm.get("hands", []) or [])
    hires_today = _int(farm.get("hires_today", 0))
    need_hires = max(0, desired_hands - current_hands)
    hire_slots = min(6 if hour <= 2 else 7, need_hires, MAX_ORDERS - len(orders))
    for _ in range(hire_slots):
        n = hires_today
        cost = HIRE_COSTS[n] if n < len(HIRE_COSTS) else HIRE_COSTS[-1] * 2
        if budget - cost < reserve and day <= 3:
            break
        orders.append(["HIRE"])
        budget -= cost
        hires_today += 1
        if len(orders) >= MAX_ORDERS:
            return orders[:MAX_ORDERS]

    # 3) Staged land expansion.  Three quadrants are enough for the validated route;
    # the SE purchase is intentionally skipped because it arrives too late to repay.
    if "NE" not in unlocked and day >= 5 and budget - reserve >= 1000 and len(orders) < MAX_ORDERS:
        orders.append(["BUY_LAND"])
        budget -= 1000
        unlocked = unlocked + ["NE"]
    elif "NE" in unlocked and "SW" not in unlocked and day >= 10 and budget - reserve >= 2000 and len(orders) < MAX_ORDERS:
        orders.append(["BUY_LAND"])
        budget -= 2000
        unlocked = unlocked + ["SW"]

    # 4) Livestock capital.  Diversify against the opponent and current relative prices.
    cow_target, sheep_target = _desired_species(obs, farm, private, opponent, day, unlocked)
    cow_have = _count_all_owned_animal(private, farm, "COW")
    sheep_have = _count_all_owned_animal(private, farm, "SHEEP")
    # Add at most 3 animals per turn to avoid a feed/workload shock.
    purchase_plan = []
    for animal, target, have in (("COW", cow_target, cow_have), ("SHEEP", sheep_target, sheep_have)):
        missing = max(0, target - have)
        if missing:
            purchase_plan.append((missing, animal))
    purchase_plan.sort(reverse=True)
    animal_room = 3
    for missing, animal in purchase_plan:
        if len(orders) >= MAX_ORDERS or animal_room <= 0:
            break
        unit_cost = ANIMALS[animal]["cost"]
        # Include two days of feed in the capital hurdle.
        wheat_price = max(1, _int(prices.get("WHEAT", 25), 25))
        affordable = int(max(0, budget - reserve) // max(1, unit_cost + 2 * wheat_price))
        n = min(missing, animal_room, affordable)
        if n > 0:
            orders.append(["BUY_ANIMAL", animal, n])
            budget -= n * unit_cost
            animal_room -= n

    # 5) Feed runway.  Own wheat production lowers this over time, but animal survival
    # has priority over speculative seed purchases.
    placed = _placed_animal_count(farm)
    carried_wheat = sum(_int(_dict(inv).get("WHEAT", 0)) for inv in inventories)
    wheat_stock = _int(shed.get("WHEAT", 0)) + carried_wheat
    wheat_goal = max(10, placed + 8)
    wheat_need = max(0, wheat_goal - wheat_stock)
    if wheat_need > 0 and len(orders) < MAX_ORDERS:
        wp = max(1, _int(prices.get("WHEAT", 25), 25))
        # If wheat is scarce, still secure one full day before paying scarcity premium.
        hard_need = max(0, placed + 2 - wheat_stock)
        cap = 12 if wp <= 45 else max(4, hard_need)
        affordable = int(max(0, budget - reserve) // wp)
        n = min(wheat_need, cap, affordable)
        if n > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", n])
            budget -= n * wp

    # 6) Seeds.  Opening composition gets first claim; later purchases are role-derived.
    needs = _role_seed_needs(obs, farm, private, opponent)
    # Day 0/1 opening is exact even before every planting slot is visible as a deficit.
    if day <= 1:
        opening = {"MELON": OPENING_MELONS, "WHEAT": OPENING_WHEAT, "CARROT": OPENING_CARROTS}
        seeds = _dict(private.get("seeds", {}))
        scan = _scan_farm(farm)
        for crop, target in opening.items():
            needs[crop] = max(needs.get(crop, 0), target - _int(seeds.get(crop, 0)) - _int(scan["plants"].get(crop, 0)))

    seed_order = ("MELON", "STRAWBERRY", "WHEAT", "CARROT", "TOMATO")
    for crop in seed_order:
        if len(orders) >= MAX_ORDERS:
            break
        need = max(0, _int(needs.get(crop, 0)))
        if need <= 0:
            continue
        if crop == "MELON" and day > 16:
            continue
        if crop == "STRAWBERRY" and day > 18:
            continue
        if crop == "TOMATO" and day > 17:
            continue
        cap = 9 if crop == "MELON" else (8 if crop == "STRAWBERRY" else 10)
        n = min(need, cap)
        unit = CROPS[crop]["seed"]
        affordable = int(max(0, budget - reserve) // unit)
        n = min(n, affordable)
        if n > 0:
            orders.append(["BUY_SEED", crop, n])
            budget -= n * unit

    return orders[:MAX_ORDERS]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _agent_impl(obs):
    player = _int(_get(obs, "player", 0))
    farms = list(_get(obs, "farms", []) or [])
    if not (0 <= player < len(farms)):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = _dict(farms[player])
    private = _dict(_get(obs, "private", {}))
    opponent = _opponent_state(obs, player)
    tasks = _build_tasks(obs, farm, private, opponent)
    farmer_action, hands_actions, projected_shed = _assign_units(obs, farm, private, tasks)
    market = _market_orders(obs, farm, private, opponent, projected_shed)
    return {"farmer": farmer_action, "hands": hands_actions, "market": market}


def agent(obs):
    """Kaggle entry point.  Any unexpected observation falls back to legal PASS."""
    try:
        return _agent_impl(obs)
    except Exception:
        farms = list(_get(obs, "farms", []) or [])
        player = _int(_get(obs, "player", 0))
        hands = 0
        if 0 <= player < len(farms):
            hands = len(_dict(farms[player]).get("hands", []) or [])
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(hands)], "market": []}
