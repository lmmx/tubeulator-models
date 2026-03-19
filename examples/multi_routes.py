# experiments/test_routes.py
from tubeulator_models import TubeRouter


router = TubeRouter.from_pretrained("permutans/tube-nexthop-policy")

TEST_PAIRS = [
    ("Angel", "Paddington"),
    ("Bethnal Green Underground", "Victoria Underground"),
    ("Clapham Common", "Liverpool Street"),
    ("Notting Hill Gate", "London Bridge"),
    ("Barons Court", "Old Street"),
    ("Brixton", "Canary Wharf"),
]

for origin, dest in TEST_PAIRS:
    print(f"\n{'=' * 60}")
    print(f"{origin} → {dest}")
    print(f"{'=' * 60}")

    routes = router.routes(origin, dest, n=5)

    for i, route in enumerate(routes):
        print(
            f"\n  Route {i + 1}: {route.total_minutes:.1f} min · "
            f"{', '.join(route.lines_used)} · "
            f"{route.n_transfers} transfers"
        )
        for step in route.steps:
            if step.is_transfer:
                print(f"    ↳ [{step.line}] {step.station}")
            elif step.line:
                print(f"    → [{step.line}] {step.station}")
            else:
                print(f"    {step.station}")
