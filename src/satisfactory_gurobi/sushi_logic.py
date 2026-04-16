import dataclasses
from dataclasses import asdict, dataclass
from collections import defaultdict
import json
from math import ceil
from pathlib import Path

from satisfactory_gurobi.parse_game_file import (
    Buildable, Form, Item, NO_CONTAINER_ITEMS, Recipe, parse_game_file, _EnumEncoder
)


@dataclass
class SushiRecipe:
    """Like Recipe, but with ingredients/products split into belt and other lanes."""
    id: str
    display_name: str
    belt_ingredients: dict[str, float]
    belt_products: dict[str, float]
    other_ingredients: dict[str, float]
    other_products: dict[str, float]
    duration: float
    produced_in: str | None
    power_consumption: float
    packaging: bool
    raw: bool
    converter: bool
    somersloop_cost: int

TARGET_RECIPE_IDS = [
    "Recipe_SpaceElevatorPart_9_C",
    "Recipe_SpaceElevatorPart_10_C",
    "Recipe_SpaceElevatorPart_11_C",
    "Recipe_SpaceElevatorPart_12_C",
]


def prepare_recipes_for_sushi(
    items: list[Item],
    recipes: list[Recipe],
    buildables: list[Buildable],
    target_recipe_ids: list[str] | None = None,
    ingredient_mult: float = 1.0,
    power_mult: float = 1.0,
    include_sloops: bool = False,
) -> tuple[list[Item], list[SushiRecipe]]:
    if target_recipe_ids is None:
        target_recipe_ids = TARGET_RECIPE_IDS

    item_form = {item.id: item.form for item in items}

    # -------------------------------------------------------------------------
    # Step 1: Filter to recipes reachable backwards from target recipes
    # -------------------------------------------------------------------------
    produces: dict[str, list[Recipe]] = defaultdict(list)
    for r in recipes:
        for item_id in r.products:
            produces[item_id].append(r)

    recipe_by_id = {r.id: r for r in recipes}
    kept: dict[str, Recipe] = {}
    queue: list[str] = []

    for rid in target_recipe_ids:
        if rid in recipe_by_id:
            r = recipe_by_id[rid]
            kept[rid] = r
            queue.extend(r.ingredients)

    visited_items: set[str] = set(queue)
    while queue:
        item_id = queue.pop()
        for r in produces.get(item_id, []):
            if r.id not in kept:
                kept[r.id] = r
                for ing in r.ingredients:
                    if ing not in visited_items:
                        visited_items.add(ing)
                        queue.append(ing)

    # -------------------------------------------------------------------------
    # Step 1b: Forward filter - remove recipes whose ingredients can't be produced
    # -------------------------------------------------------------------------
    stable = False
    while not stable:
        producible = {item_id for r in kept.values() for item_id in r.products}
        new_kept = {rid: r for rid, r in kept.items() if all(ing in producible for ing in r.ingredients)}
        stable = len(new_kept) == len(kept)
        kept = new_kept

    filtered = list(kept.values())

    # -------------------------------------------------------------------------
    # Step 2: Apply ingredient_mult and power_mult
    # -------------------------------------------------------------------------
    def scale_recipe(r: Recipe) -> Recipe:
        new_ingredients = r.ingredients
        if not r.packaging and ingredient_mult != 1.0:
            new_ingredients = {
                k: float(ceil(v * ingredient_mult))
                for k, v in r.ingredients.items()
            }
        new_power = (
            float(ceil(r.power_consumption * power_mult))
            if r.power_consumption is not None
            else None
        )
        return dataclasses.replace(r, ingredients=new_ingredients, power_consumption=new_power)

    scaled = [scale_recipe(r) for r in filtered]

    # -------------------------------------------------------------------------
    # Step 3: Slooped variants
    # -------------------------------------------------------------------------
    building_sloop = {b.id: b.somersloop_cost for b in buildables}
    result = list(scaled)

    if include_sloops:
        slooped = []
        for r in scaled:
            slots = building_sloop.get(r.produced_in, 0) if r.produced_in else 0
            if slots > 0:
                slooped.append(dataclasses.replace(
                    r,
                    id=r.id + "_S",
                    products={k: v * 2 for k, v in r.products.items()},
                    power_consumption=(
                        float(ceil(r.power_consumption * 4))
                        if r.power_consumption is not None else None
                    ),
                ))
        result.extend(slooped)

    # -------------------------------------------------------------------------
    # Step 4: Auto-package/unpackage fluids
    # -------------------------------------------------------------------------
    def is_packagable_fluid(item_id: str) -> bool:
        return (
            item_form.get(item_id) in (Form.LIQUID, Form.GAS)
            and item_id not in NO_CONTAINER_ITEMS
        )

    # Build lookup: fluid_id -> packaging recipe (fluid+container → packaged)
    #               fluid_id -> unpackaging recipe (packaged → fluid+container)
    package_recipe_for: dict[str, Recipe] = {}
    unpackage_recipe_for: dict[str, Recipe] = {}

    for r in result:
        if not r.packaging:
            continue
        for item_id in r.ingredients:
            if is_packagable_fluid(item_id):
                package_recipe_for[item_id] = r
        for item_id in r.products:
            if is_packagable_fluid(item_id):
                unpackage_recipe_for[item_id] = r

    final: list[Recipe] = []
    for r in result:
        # Packaging recipes are baked in; remove as standalone recipes
        if r.packaging:
            continue

        new_ingredients = dict(r.ingredients)
        new_products = dict(r.products)
        extra_power = 0.0

        # Fluid ingredients → merge in corresponding unpackage recipe
        for item_id in [i for i in list(new_ingredients) if is_packagable_fluid(i)]:
            u = unpackage_recipe_for.get(item_id)
            if u is None:
                continue
            a_F = new_ingredients.pop(item_id)
            u_F = u.products[item_id]
            # amount_scale = unpackage cycles needed per R cycle
            amount_scale = a_F / u_F
            for ing, amt in u.ingredients.items():
                new_ingredients[ing] = new_ingredients.get(ing, 0.0) + amount_scale * amt
            for prod, amt in u.products.items():
                if prod != item_id:
                    new_products[prod] = new_products.get(prod, 0.0) + amount_scale * amt
            if u.power_consumption is not None:
                # machine_ratio accounts for duration difference between U and R
                machine_ratio = amount_scale * u.duration / r.duration
                extra_power += machine_ratio * u.power_consumption

        # Fluid products → merge in corresponding package recipe
        for item_id in [p for p in list(new_products) if is_packagable_fluid(p)]:
            p_rec = package_recipe_for.get(item_id)
            if p_rec is None:
                continue
            a_F = new_products.pop(item_id)
            p_F = p_rec.ingredients[item_id]
            amount_scale = a_F / p_F
            for ing, amt in p_rec.ingredients.items():
                if ing != item_id:
                    new_ingredients[ing] = new_ingredients.get(ing, 0.0) + amount_scale * amt
            for prod, amt in p_rec.products.items():
                new_products[prod] = new_products.get(prod, 0.0) + amount_scale * amt
            if p_rec.power_consumption is not None:
                machine_ratio = amount_scale * p_rec.duration / r.duration
                extra_power += machine_ratio * p_rec.power_consumption

        final.append(dataclasses.replace(
            r,
            ingredients=new_ingredients,
            products=new_products,
            power_consumption=(r.power_consumption or 0.0) + extra_power,
        ))


    # -------------------------------------------------------------------------
    # Step 5: Split ingredients/products into belt vs. other lanes
    # -------------------------------------------------------------------------
    def to_sushi_recipe(r: Recipe) -> SushiRecipe:
        belt_ingredients = {k: v for k, v in r.ingredients.items() if k not in NO_CONTAINER_ITEMS}
        other_ingredients = {k: v for k, v in r.ingredients.items() if k in NO_CONTAINER_ITEMS}
        belt_products = {k: v for k, v in r.products.items() if k not in NO_CONTAINER_ITEMS}
        other_products = {k: v for k, v in r.products.items() if k in NO_CONTAINER_ITEMS}
        return SushiRecipe(
            id=r.id,
            display_name=r.display_name,
            belt_ingredients=belt_ingredients,
            belt_products=belt_products,
            other_ingredients=other_ingredients,
            other_products=other_products,
            duration=r.duration,
            produced_in=r.produced_in,
            power_consumption=r.power_consumption or 0.0,
            packaging=r.packaging,
            raw=r.raw,
            converter=r.converter,
            somersloop_cost=r.somersloop_cost,
        )

    sushi_final = [to_sushi_recipe(r) for r in final]

    used_item_ids = {item_id for r in sushi_final for item_id in (*r.belt_ingredients, *r.belt_products, *r.other_ingredients, *r.other_products)}
    filtered_items = [item for item in items if item.id in used_item_ids]

    return filtered_items, sushi_final

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

if __name__ == "__main__":
    buildables, items, recipes = parse_game_file(DATA_DIR / "en-GB.json")
    items, recipes = prepare_recipes_for_sushi(items, recipes, buildables, power_mult=0.25, ingredient_mult=0.25, include_sloops=True)

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "sushi-items.json", "w", encoding="utf-8") as f:
        json.dump([asdict(i) for i in items], f, indent=2, cls=_EnumEncoder)

    with open(OUTPUT_DIR / "sushi-recipes.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in recipes], f, indent=2, cls=_EnumEncoder)

    print(f"Written {len(items)} items, {len(recipes)} recipes to {OUTPUT_DIR}")

