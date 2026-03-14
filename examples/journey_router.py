from tubeulator_models import TubeRouter


router = TubeRouter.from_pretrained("permutans/tube-nexthop-policy")
route = router.route("West Ham Underground", "Shoreditch")
print(route)
