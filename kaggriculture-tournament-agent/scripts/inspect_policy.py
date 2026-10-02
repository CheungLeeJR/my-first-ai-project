from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent


def main():
    opening = agent._opening_maps(10)
    long_term = agent._long_term_site_sets(10)

    print("Opening layout")
    print("  melon:", len(opening[0]))
    print("  wheat:", len(opening[1]))
    print("  carrot:", len(opening[2]))
    print()

    print("Long-term role layout")
    print("  strawberry:", len(long_term[0]))
    print("  melon:", len(long_term[1]))
    print("  feed wheat:", len(long_term[2]))
    print("  flex:", len(long_term[3]))
    print("  animal sites:", len(agent.ANIMAL_SITES))
    print()

    print("Sample market curves around equilibrium inventory")
    for item, params in agent.BASE_MARKET_PARAMS.items():
        i0 = params["I0"]
        values = [
            agent._market_price(item, i0 - 100, {}),
            agent._market_price(item, i0, {}),
            agent._market_price(item, i0 + 100, {}),
        ]
        print(f"  {item:12s}: I0-100={values[0]:4d}  I0={values[1]:4d}  I0+100={values[2]:4d}")


if __name__ == "__main__":
    main()
