import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from satisfactory_gurobi.parse_game_file import parse_game_file
from satisfactory_gurobi.sushi_logic import SushiRecipe, prepare_recipes_for_sushi

DATA_DIR = Path(__file__).parent.parent.parent / "data"

BELT_CAPACITY = 1200

def main() -> None:
    buildables, items, recipes = parse_game_file(DATA_DIR / "en-GB.json")
    items, recipes = prepare_recipes_for_sushi(items, recipes, buildables, power_mult=1, ingredient_mult=0.25, include_sloops=False)

    # Hacky code starts...
    # Remove slooped recipes for now - we can think about slooping later
    recipes = [r for r in recipes if not r.id.endswith("_S")]

    # Remove converter recipes for now
    recipes = [r for r in recipes if not r.converter]
    # ... hacky code ends

    # Index sets
    item_ids = [item.id for item in items]
    recipe_ids = [recipe.id for recipe in recipes]

    # Parameters
    value = {item.id: item.value for item in items}

    belt_consumes = {
        (recipe.id, i_id): i_amount / recipe.duration * 60
        for recipe in recipes
        for (i_id, i_amount) in recipe.belt_ingredients.items()
    }
    belt_produces = {
        (recipe.id, i_id): i_amount / recipe.duration * 60
        for recipe in recipes
        for (i_id, i_amount) in recipe.belt_products.items()
    }
    other_consumes = {
        (recipe.id, i_id): i_amount / recipe.duration * 60
        for recipe in recipes
        for (i_id, i_amount) in recipe.other_ingredients.items()
    }
    other_produces = {
        (recipe.id, i_id): i_amount / recipe.duration * 60
        for recipe in recipes
        for (i_id, i_amount) in recipe.other_products.items()
    }

    def net_belt(r_id: str, i_id: str) -> float:
        return belt_produces.get((r_id, i_id), 0.0) - belt_consumes.get((r_id, i_id), 0.0)

    def net_other(r_id: str, i_id: str) -> float:
        return other_produces.get((r_id, i_id), 0.0) - other_consumes.get((r_id, i_id), 0.0)

    model = gp.Model("satisfactory")

    # Decision variables: how many of each recipe is run
    scale = model.addVars(recipe_ids, lb=0.0, ub=10000000.0, name="scale")

    # Constraints:
    # Initial resources output must be <= 1200
    raw_recipes_ids = [recipe.id for recipe in recipes if recipe.raw]
    model.addConstr(gp.quicksum(
        scale[r_id] * belt_produces.get((r_id, i_id), 0.0)
        for r_id in raw_recipes_ids
        for i_id in item_ids
        ) <= 1200)

    # beltItemNonnegConstr
    for i_id in item_ids:
        model.addConstr(gp.quicksum(scale[r_id] * net_belt(r_id, i_id) for r_id in recipe_ids) >= 0, name=f"beltNonneg[{i_id}]")

    # otherItemBalanceConstr
    for i_id in item_ids:
        model.addConstr(
            gp.quicksum(scale[r_id] * net_other(r_id, i_id) for r_id in recipe_ids) == 0,
            name=f"otherBalance[{i_id}]"
        )

    # Objective: maximize total value of items on belt at end
    model.setObjective(
        gp.quicksum(value[i_id] * scale[r_id] * net_belt(r_id, i_id) for r_id in recipe_ids for i_id in item_ids),
        GRB.MAXIMIZE
    )
    # model.setObjective(scale["Recipe_SpaceElevatorPart_12_C"], GRB.MAXIMIZE)

    model.update()
    print(f"Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print(f"Optimal value: {model.ObjVal:.2f}")
        final_scales = []
        for r_id in recipe_ids:
            v = scale[r_id].X
            if v > 1e-6:
                final_scales.append((r_id, v))
        final_scales.sort(key=lambda t: t[1])
        [print(f"  {r_id} x{v:.4f}") for (r_id, v) in final_scales]

        print(f"Final items on belt")
        final_items = []
        for i_id in item_ids:
            total = 0
            for r_id in recipe_ids:
                total += scale[r_id].X * net_belt(r_id, i_id)
            if total > 1e-6:
                final_items.append((i_id, total))
        final_items.sort(key=lambda t: t[1])
        [print(f"    {i_id} x{total} (value {value[i_id]}) (total_value {value[i_id] * total})") for (i_id, total) in final_items]
    else:
        print(f"No optimal solution found (status {model.Status})")
    
    # Now find the optimal ordering of these recipe-scale pairs
    solve_sequencing(final_scales, item_ids, net_belt, recipes)


def solve_sequencing(
    final_scales: list[tuple[str, float]],
    item_ids: list[str],
    net_belt,
    recipes: list[SushiRecipe],
) -> None:
    n = len(final_scales)
    print(f"Solving a sequencing problem with {n} recipes")

    # Precompute net belt contribution per entry per item (known constants)
    delta = {
        (e, i_id): v * net_belt(r_id, i_id)
        for e, (r_id, v) in enumerate(final_scales)
        for i_id in item_ids
    }

    # Only track items that are actually affected by at least one entry
    active_item_ids = [
        i_id for i_id in item_ids
        if any(abs(delta[e, i_id]) > 1e-9 for e in range(n))
    ]

    # Greedy search: at each step pick the feasible entry that minimises peak belt usage
    belt = {i_id: 0.0 for i_id in active_item_ids}
    remaining = list(range(n))
    sequence: list[tuple[str, float]] = []
    peak = 0.0
    belt_load_at_slot = [0.0]

    # Fix the starting order:
    starting_order = [
        "Extract_Desc_Water_C",
        "Extract_Desc_LiquidOil_C",
        "Extract_Desc_NitrogenGas_C",
        "Mine_Desc_OreIron_C",
        "Mine_Desc_Sulfur_C",
        "Mine_Desc_OreGold_C",
        "Mine_Desc_OreCopper_C",
        "Mine_Desc_SAM_C",
        "Mine_Desc_RawQuartz_C",
        "Mine_Desc_OreBauxite_C",
        "Mine_Desc_Stone_C"
        ]
    for e in [r for r in remaining if final_scales[r][0] in starting_order]: 
        for i_id in active_item_ids:
            if abs(delta[e, i_id]) > 1e-9:
                print(f"adding {e} {i_id} {delta[e, i_id]}")

            belt[i_id] += delta[e, i_id]
        belt_load_at_slot.append(sum(belt.values()))
        peak = max(peak, sum(belt.values()))
        sequence.append(final_scales[e])
        remaining.remove(e)

    # This makes some things go negative. Add them at the start - they are recycled at the end
    recycled: map[str, float] = {}
    for (i_id, amount) in belt.items():
        if amount < 0:
            belt[i_id] = 0.0
            recycled[i_id] = -amount

    # From experimentation, there's some resources which needs to loop back. The only way this can work is
    # by going off the end.
    extra_recycled = {
        "Desc_FluidCanister_C": 168.0,
        "Desc_PackagedWater_C": 1114,
        "Desc_Plastic_C": 10000.0
    }
    for (i_id, amount) in extra_recycled.items():
        belt[i_id] += amount
        recycled[i_id] = recycled.get(i_id, 0.0) + amount 

    # Rewrite belt_load_at_slot to account for recycling
    total_recycled = sum(recycled.values())
    belt_load_at_slot = [x + total_recycled for x in belt_load_at_slot]
    peak += total_recycled

    while remaining:
        best_e = None
        best_peak = float('inf')

        for e in remaining:
            new_belt = {i_id: belt[i_id] + delta[e, i_id] for i_id in active_item_ids}
            if any(v < -1e-9 for v in new_belt.values()):
                problem_items = {(k, new_belt[k]) for k in new_belt if new_belt[k] < -1e-9}
                print(f"{final_scales[e][0]}: Goes negative {problem_items}")
                continue  # infeasible placement
            print(f"Found one!\n")
            candidate_peak = sum(new_belt.values())
            if candidate_peak < best_peak:
                best_peak = candidate_peak
                best_e = e

        if best_e is None:            
            print(f"No feasible placement found at step {len(sequence) + 1} — no valid ordering exists.")
            print(f"\n{sequence}\n\n{belt}\n\n{[final_scales[e][0] for e in remaining]}")
            return

        for i_id in active_item_ids:
            belt[i_id] += delta[best_e, i_id]
        belt_load_at_slot.append(sum(belt.values()))
        peak = max(peak, sum(belt.values()))
        sequence.append(final_scales[best_e])
        remaining.remove(best_e)

    print(f"Starting belt load from recycled goods: {belt_load_at_slot[0]:.2f}")
    print(f"Peak belt usage: {peak:.2f}")
    for s, (r_id, v) in enumerate(sequence):
        print(f"  Slot {s + 1}: {r_id} x{v:.4f} (Belt load: {belt_load_at_slot[s+1]:.2f})")



if __name__ == "__main__":
    main()

