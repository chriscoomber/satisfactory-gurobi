import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from satisfactory_gurobi.parse_game_file import parse_game_file
from satisfactory_gurobi.sushi_logic import SushiRecipe, prepare_recipes_for_sushi

DATA_DIR = Path(__file__).parent.parent.parent / "data"

BELT_CAPACITY = 1200

TOLERANCE = {
    0.25: 2.76e6,
    1.0: 0
}

INGREDIENT_MULT = 0.25

POWER_MULT = 0.25

def main() -> None:
    buildables, items, recipes = parse_game_file(DATA_DIR / "en-GB.json")
    items, recipes = prepare_recipes_for_sushi(items, recipes, buildables, power_mult=POWER_MULT, ingredient_mult=INGREDIENT_MULT, include_sloops=False)

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
    # Upper bound tries to limit how much of each recipe is created
    scale = model.addVars(recipe_ids, lb=0.0, ub=(1200/max(1, sum(net_belt(r_id, i_id) for i_id in item_ids)) for r_id in recipe_ids), name="scale")

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

    # No single recipe should produce over 1200

    model.ModelSense = GRB.MAXIMIZE
    # Objective: maximize total value of items on belt at end
    model.setObjectiveN(
        gp.quicksum(value[i_id] * scale[r_id] * net_belt(r_id, i_id) for r_id in recipe_ids for i_id in item_ids),
        1,
        priority=2,
        weight=1,
        abstol=TOLERANCE[INGREDIENT_MULT], # Manual tuning
        name="value"
    )

    # Objective: Minimize the total belt load across every recipe
    model.setObjectiveN(
        gp.quicksum(scale[r_id] * net_belt(r_id, i_id) for i_id in item_ids for r_id in recipe_ids),
        2,
        priority=1,
        weight=-1,
        name="beltLoad"
    )

    # model.setObjective(scale["Recipe_SpaceElevatorPart_12_C"], GRB.MAXIMIZE)

    model.update()
    print(f"Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        model.params.ObjNumber = 1
        print(f"Optimal value: {model.ObjPassNObjVal:.2f}")
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
    peak_belt_usage = solve_sequencing(final_scales, item_ids, net_belt, recipes, value)
    print(f"Overall best value/min = {model.ObjPassNObjVal * 1200/peak_belt_usage:.2f}")


BEAM_WIDTH = 10


def solve_sequencing(
    final_scales: list[tuple[str, float]],
    item_ids: list[str],
    net_belt,
    recipes: list[SushiRecipe],
    value: dict[str, float],
) -> float:
    """
    TODO: This function doesn't appear to work properly, or if it does, the above LP
    optimization produces a poor result for reducing belt load.
    """
    # Pre-sort: lowest net value-added recipes first.
    # Net value = sum over items of (value[item] * net_belt contribution).
    # Low-value recipes tend to be raw-material producers that need to come early anyway;
    # this gives the beam search a better starting order to work from.
    final_scales = sorted(
        final_scales,
        key=lambda rv: sum(value.get(i_id, 0.0) * net_belt(rv[0], i_id) for i_id in item_ids),
    )

    n = len(final_scales)
    print(f"Solving a sequencing problem with {n} recipes")

    delta = {
        (e, i_id): v * net_belt(r_id, i_id)
        for e, (r_id, v) in enumerate(final_scales)
        for i_id in item_ids
    }

    active_item_ids = [
        i_id for i_id in item_ids
        if any(abs(delta[e, i_id]) > 1e-9 for e in range(n))
    ]
    m = len(active_item_ids)

    # 2D list for fast inner-loop access: d[e][k] = net belt contribution of entry e to item k
    d = [[delta[e, i_id] for i_id in active_item_ids] for e in range(n)]

    def _peak(order: list[int]) -> float:
        """Exact peak belt usage for a given ordering of entry indices.
        Peak = sum_i(start[i]) + max_k(net_total[k])
        where start[i] = max(0, -min_ever_cum[i]) and net_total[k] = sum of cum_belt at step k.
        """
        cum = [0.0] * m
        lo = [0.0] * m   # per-item minimum cumulative (determines starting inventory)
        max_net = 0.0     # tracks max(net_totals), initialised to net_totals[0] = 0
        for e in order:
            net = 0.0
            for k in range(m):
                cum[k] += d[e][k]
                if cum[k] < lo[k]:
                    lo[k] = cum[k]
                net += cum[k]
            if net > max_net:
                max_net = net
        S = sum(-v for v in lo if v < 0)
        return S + max_net

    def _score(cum: list, lo: list, neg: list, max_net: float, e: int) -> float:
        """Estimated final peak if entry e is placed next.
        Uses worst-case future negatives (all consumers before producers for each item)
        as a lower bound on final starting inventory.
        """
        S = 0.0
        net = 0.0
        for k in range(m):
            tc = cum[k] + d[e][k]
            tm = lo[k] if lo[k] < tc else tc
            neg_excl = neg[k] - (d[e][k] if d[e][k] < 0.0 else 0.0)
            em = tm if tm < tc + neg_excl else tc + neg_excl
            if em < 0.0:
                S -= em
            net += tc
        return S + (net if net > max_net else max_net)

    def _advance(cum: list, lo: list, neg: list, max_net: float, e: int):
        """Advance state by placing entry e. Returns (new_cum, new_lo, new_neg, new_max_net)."""
        nc = [cum[k] + d[e][k] for k in range(m)]
        nl = [lo[k] if lo[k] < nc[k] else nc[k] for k in range(m)]
        nn = [neg[k] - (d[e][k] if d[e][k] < 0.0 else 0.0) for k in range(m)]
        net = sum(nc)
        return nc, nl, nn, (net if net > max_net else max_net)

    # Fixed starting order for raw resources (placed before beam search begins)
    starting_order = {
        "Extract_Desc_Water_C", "Extract_Desc_LiquidOil_C", "Extract_Desc_NitrogenGas_C",
        "Mine_Desc_OreIron_C", "Mine_Desc_Sulfur_C", "Mine_Desc_OreGold_C",
        "Mine_Desc_OreCopper_C", "Mine_Desc_SAM_C", "Mine_Desc_RawQuartz_C",
        "Mine_Desc_OreBauxite_C", "Mine_Desc_Stone_C", "Mine_Desc_Coal_C",
    }
    prefix = [e for e in range(n) if final_scales[e][0] in starting_order]
    free   = [e for e in range(n) if final_scales[e][0] not in starting_order]

    cum0  = [0.0] * m
    lo0   = [0.0] * m
    neg0  = [sum(d[e][k] for e in range(n) if d[e][k] < 0.0) for k in range(m)]
    mnet0 = 0.0
    for e in prefix:
        cum0, lo0, neg0, mnet0 = _advance(cum0, lo0, neg0, mnet0, e)

    # -------------------------------------------------------------------------
    # Beam search over the free entries
    # Each beam state: (order, remaining, cum, lo, neg, max_net)
    # -------------------------------------------------------------------------
    # State tuple indices
    _ORD, _REM, _CUM, _LO, _NEG, _MN = 0, 1, 2, 3, 4, 5

    beam: list[tuple] = [(prefix[:], free[:], cum0, lo0, neg0, mnet0)]

    while beam[0][_REM]:
        candidates: list[tuple] = []
        for bi, state in enumerate(beam):
            cum, lo, neg, max_net = state[_CUM], state[_LO], state[_NEG], state[_MN]
            for e in state[_REM]:
                sc = _score(cum, lo, neg, max_net, e)
                candidates.append((sc, bi, e, state))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        new_beam: list[tuple] = []
        for sc, bi, e, state in candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break
            nc, nl, nn, nm = _advance(state[_CUM], state[_LO], state[_NEG], state[_MN], e)
            new_beam.append((
                state[_ORD] + [e],
                [x for x in state[_REM] if x != e],
                nc, nl, nn, nm,
            ))
        beam = new_beam

    best_order: list[int] = min((s[_ORD] for s in beam), key=_peak)
    print(f"Beam search peak: {_peak(best_order):.2f}. Running 2-opt local search...")

    # -------------------------------------------------------------------------
    # 2-opt local search: try all pairwise swaps within the free portion only.
    # The prefix (starting_order entries) stays fixed at the front.
    # -------------------------------------------------------------------------
    free_start = len(prefix)
    current_peak = _peak(best_order)
    improved = True
    while improved:
        improved = False
        for i in range(free_start, len(best_order)):
            for j in range(i + 1, len(best_order)):
                best_order[i], best_order[j] = best_order[j], best_order[i]
                p = _peak(best_order)
                if p < current_peak - 1e-6:
                    current_peak = p
                    improved = True
                else:
                    best_order[i], best_order[j] = best_order[j], best_order[i]
            if improved:
                break  # restart the outer loop

    # -------------------------------------------------------------------------
    # Report results
    # -------------------------------------------------------------------------
    cum = [0.0] * m
    lo  = [0.0] * m
    net_totals = [0.0]
    for e in best_order:
        for k in range(m):
            cum[k] += d[e][k]
            if cum[k] < lo[k]:
                lo[k] = cum[k]
        net_totals.append(sum(cum))

    starting_inventory = {active_item_ids[k]: max(0.0, -lo[k]) for k in range(m)}
    S = sum(starting_inventory.values())
    peak = S + max(net_totals)
    net_total_at_slot = [S + t for t in net_totals]

    si_nonzero = {i_id: amt for i_id, amt in starting_inventory.items() if amt > 1e-6}
    if si_nonzero:
        print(f"Starting inventory needed ({len(si_nonzero)} items):")
        for i_id, amt in sorted(si_nonzero.items(), key=lambda kv: -kv[1]):
            print(f"  {i_id}: {amt:.4f}/min")
    print(f"Peak belt usage: {peak:.2f}")
    for s, e in enumerate(best_order):
        r_id, v = final_scales[e]
        print(f"  Slot {s + 1}: {r_id} x{v:.4f} (belt load: {net_total_at_slot[s + 1]:.2f})")

    return peak


if __name__ == "__main__":
    main()

