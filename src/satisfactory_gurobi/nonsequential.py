import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from satisfactory_gurobi import parse

DATA_DIR = Path(__file__).parent.parent.parent / "data"

BELT_CAPACITY = 1200

def main() -> None:
    items = parse.parse_items(DATA_DIR / "sushi_items.json")
    recipes = parse.parse_recipes(DATA_DIR / "sushi_recipes.json")

    # Hacky code starts...
    # Remove slooped recipes for now - we can think about slooping later
    recipes = [r for r in recipes if not r.id.endswith("_S")]

    # banned_recipe_ids = ["MineOreCopper", "MineOreIron", "MineStone", "MineSulfur", "MineCoal", "MineRawQuartz", "MineOreBauxite", "MineOreGold", "ExtractNitrogenGas", "RecycleCanister", "RecycleTank"]
    # Remove free canister/tank recipes
    banned_recipe_ids = ["RecycleCanister", "RecycleTank"]
    recipes = [r for r in recipes if not r.id in banned_recipe_ids]

    # Filter out any recipes which can't be made from these raw resources, plus the seed iron ore for the converters
    known_items = set(["Desc_SAM_C", "Desc_PackagedOil_C", "Desc_PackagedWater_C", "Desc_OreIron_C", "Desc_GasTank_C", "Desc_FluidCanister_C"])
    recipes_to_check = [r for r in recipes]
    filtered_recipes = []
    steps_since_last_improvement = 0
    while steps_since_last_improvement < 250 and len(recipes_to_check) > 0:
        r = recipes_to_check.pop()
        if all(i in set().union(known_items).union([i.item_id for i in r.belt_products]).union([i.item_id for i in r.other_products]) for i in set().union([i.item_id for i in r.belt_ingredients]).union([i.item_id for i in r.other_ingredients])):
            steps_since_last_improvement = 0
            known_items.update([i.item_id for i in r.belt_products], [i.item_id for i in r.other_products])
            filtered_recipes.append(r)
        else:
            recipes_to_check.insert(0, r)
            steps_since_last_improvement += 1
    recipes = filtered_recipes
    print(f"Using recipes: \n{"".join([f"    {r.id}\n" for r in recipes])}")
    # ... hacky code ends

    # Index sets
    item_ids = [item.id for item in items]
    recipe_ids = [recipe.id for recipe in recipes]

    # Parameters
    value = {item.id: item.sink_points for item in items}

    belt_consumes = {
        (recipe.id, ia.item_id): ia.amount / recipe.duration * 60
        for recipe in recipes
        for ia in recipe.belt_ingredients
    }
    belt_produces = {
        (recipe.id, ia.item_id): ia.amount / recipe.duration * 60
        for recipe in recipes
        for ia in recipe.belt_products
    }
    other_consumes = {
        (recipe.id, ia.item_id): ia.amount / recipe.duration * 60
        for recipe in recipes
        for ia in recipe.other_ingredients
    }
    other_produces = {
        (recipe.id, ia.item_id): ia.amount / recipe.duration * 60
        for recipe in recipes
        for ia in recipe.other_products
    }

    def net_belt(r_id: str, i_id: str) -> float:
        return belt_produces.get((r_id, i_id), 0.0) - belt_consumes.get((r_id, i_id), 0.0)

    def net_other(r_id: str, i_id: str) -> float:
        return other_produces.get((r_id, i_id), 0.0) - other_consumes.get((r_id, i_id), 0.0)

    model = gp.Model("satisfactory")

    # Decision variables: how many of each recipe is run
    scale = model.addVars(recipe_ids, lb=0.0, name="scale")

    # Constraints:
    # Initial resources (SAM Ore + Oil + Water) must be <= 1200
    model.addConstr(gp.quicksum(
        scale[r_id]
        for r_id in ["MineSAM", "ExtractLiquidOil", "ExtractWater", "MineOreCopper", "MineOreIron", "MineStone", "MineSulfur", "MineCoal", "MineRawQuartz", "MineOreBauxite", "MineOreGold", "ExtractNitrogenGas"]
        ) <= 1200/60)

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
    solve_sequencing(final_scales, item_ids, net_belt)


def solve_sequencing(
    final_scales: list[tuple[str, float]],
    item_ids: list[str],
    net_belt,
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

    # Some products we allow to go negative
    active_item_ids = [id for id in active_item_ids if id not in ["Desc_GasTank_C", "Desc_FluidCanister_C", "Desc_DarkEnergy_C", "Desc_QuantumEnergy_C", "Desc_PackagedWater_C"]]

    # Greedy search: at each step pick the feasible entry that minimises peak belt usage
    belt = {i_id: 0.0 for i_id in active_item_ids}
    remaining = list(range(n))
    sequence = []
    peak = 0.0

    # Fix the starting order:
    starting_order = ["ExtractWater", "ExtractNitrogenGas", "MineOreIron", "MineSulfur", "MineOreGold", "MineOreCopper", "MineSAM", "MineRawQuartz", "MineOreBauxite", "ExtractLiquidOil"]
    for e in [r for r in remaining if final_scales[r][0] in starting_order]: 
        for i_id in active_item_ids:
            belt[i_id] += delta[e, i_id]
        peak = max(peak, sum(belt.values()))
        sequence.append(final_scales[e])
        remaining.remove(e)

    while remaining:
        best_e = None
        best_peak = float('inf')

        for e in remaining:
            new_belt = {i_id: belt[i_id] + delta[e, i_id] for i_id in active_item_ids}
            if any(v < -1e-9 for v in new_belt.values()):
                # problem_items = {(k, new_belt[k]) for k in new_belt if new_belt[k] < -1e-9}
                # print(f"{final_scales[e][0]}: Goes negative {problem_items}")
                continue  # infeasible placement
            candidate_peak = sum(new_belt.values())
            if candidate_peak < best_peak:
                best_peak = candidate_peak
                best_e = e

        if best_e is None:            
            print(f"No feasible placement found at step {len(sequence) + 1} — no valid ordering exists.")
            # print(f"\n{sequence}\n\n{belt}\n\n{[final_scales[e][0] for e in remaining]}")
            return

        for i_id in active_item_ids:
            belt[i_id] += delta[best_e, i_id]
        peak = max(peak, sum(belt.values()))
        sequence.append(final_scales[best_e])
        remaining.remove(best_e)
        # print("found one!\n\n")

    print(f"Peak belt usage: {peak:.2f}")
    for s, (r_id, v) in enumerate(sequence):
        print(f"  Slot {s + 1}: {r_id} x{v:.4f}")



if __name__ == "__main__":
    main()

