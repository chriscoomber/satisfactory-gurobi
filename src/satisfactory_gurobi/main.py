import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from satisfactory_gurobi import parse

DATA_DIR = Path(__file__).parent.parent.parent / "data"

SLOT_LIMIT = 10
BELT_CAPACITY = 1200


def main() -> None:
    items = parse.parse_items(DATA_DIR / "sushi_items.json")
    recipes = parse.parse_recipes(DATA_DIR / "sushi_recipes.json")

    # Hacky code starts...
    # Remove slooped recipes for now - we can think about slooping later
    recipes = [r for r in recipes if not r.id.endswith("_S")]

    # Only go up to phase 4
    base_recipe_ids = ["Recipe_SpaceElevatorPart_9_C", "Recipe_SpaceElevatorPart_8_C", "Recipe_SpaceElevatorPart_7_C", "Recipe_SpaceElevatorPart_6_C"]
    base_recipes = [r for r in recipes if r.id in base_recipe_ids]
    recipes = [r for r in recipes if r not in base_recipes]
    known_items = set()
    for r in base_recipes:
        known_items.update([i.item_id for i in r.belt_ingredients], [i.item_id for i in r.other_ingredients])

    filtered_recipes = base_recipes
    steps_since_last_improvement = 0
    while steps_since_last_improvement < 250 and len(recipes) > 0:
        r = recipes.pop()
        if any(i in known_items for i in set().union([i.item_id for i in r.belt_products]).union([i.item_id for i in r.other_products])):
            steps_since_last_improvement = 0
            known_items.update([i.item_id for i in r.belt_ingredients], [i.item_id for i in r.other_ingredients])
            filtered_recipes.append(r)
        else:
            recipes.insert(0, r)
            steps_since_last_improvement += 1

    recipes = filtered_recipes
  
    # Remove conversion recipes
    recipes = [r for r in recipes if "(" not in r.display_name]
    print(f"There are: {len(recipes)} recipes")

    # ... hacky code ends

    # Index sets
    item_ids = [item.id for item in items]
    recipe_ids = [recipe.id for recipe in recipes]
    slots = list(range(1, SLOT_LIMIT + 1))

    # Parameters
    value = {item.id: item.sink_points for item in items}

    belt_consumes = {
        (recipe.id, ia.item_id): ia.amount
        for recipe in recipes
        for ia in recipe.belt_ingredients
    }
    belt_produces = {
        (recipe.id, ia.item_id): ia.amount
        for recipe in recipes
        for ia in recipe.belt_products
    }
    other_consumes = {
        (recipe.id, ia.item_id): ia.amount
        for recipe in recipes
        for ia in recipe.other_ingredients
    }
    other_produces = {
        (recipe.id, ia.item_id): ia.amount
        for recipe in recipes
        for ia in recipe.other_products
    }

    def net_belt(r_id: str, i_id: str) -> float:
        return belt_produces.get((r_id, i_id), 0.0) - belt_consumes.get((r_id, i_id), 0.0)

    def net_other(r_id: str, i_id: str) -> float:
        return other_produces.get((r_id, i_id), 0.0) - other_consumes.get((r_id, i_id), 0.0)

    model = gp.Model("satisfactory")

    # Decision variables: how many of each recipe is placed in each slot
    assign = model.addVars(recipe_ids, slots, lb=0.0, name="assign")
    
    model.update()
    print(f"Z: Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")

    # # Auxiliary variables: cumulative belt items at each slot
    # belt_culm = model.addVars(item_ids, slots, lb=-GRB.INFINITY, name="beltItemCulmAtSlot")

    # model.update()
    # print(f"A: Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")


    # # Belt cumulation definition
    # for i_id in item_ids:
    #     for s in slots:
    #         net_from_slot = gp.quicksum(assign[r_id, s] * net_belt(r_id, i_id) for r_id in recipe_ids)
    #         if s == 1:
    #             model.addConstr(belt_culm[i_id, s] == net_from_slot, name=f"beltCulm[{i_id},{s}]")
    #         else:
    #             model.addConstr(belt_culm[i_id, s] == belt_culm[i_id, s - 1] + net_from_slot, name=f"beltCulm[{i_id},{s}]")
    # model.update()
    # print(f"B: Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")


    # assignConstr: at most one recipe active per slot, via SOS1
    for s in slots:
        model.addSOS(GRB.SOS_TYPE1, [assign[r_id, s] for r_id in recipe_ids])
    model.update()
    print(f"C: Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")


    # beltItemNonnegConstr
    for i_id in item_ids:
        for s in slots:
            model.addConstr(gp.quicksum(assign[r_id, s_index] * net_belt(r_id, i_id) for r_id in recipe_ids for s_index in range(1, s+1)) >= 0, name=f"beltNonneg[{i_id},{s}]")

    # beltCapacityConstr
    for s in slots:
        model.addConstr(
            gp.quicksum(assign[r_id, s_index] * net_belt(r_id, i_id) for r_id in recipe_ids for s_index in range(1, s+1) for i_id in item_ids) <= BELT_CAPACITY,
            name=f"beltCapacity[{s}]"
        )

    # otherItemBalanceConstr
    for i_id in item_ids:
        model.addConstr(
            gp.quicksum(assign[r_id, s] * net_other(r_id, i_id) for r_id in recipe_ids for s in slots) == 0,
            name=f"otherBalance[{i_id}]"
        )

    # Objective: maximize total value of items on belt at final slot
    model.setObjective(
        gp.quicksum(value[i_id] * assign[r_id, s_index] * net_belt(r_id, i_id) for r_id in recipe_ids for s_index in slots for i_id in item_ids),
        GRB.MAXIMIZE
    )

    model.update()
    print(f"Model size: {model.NumVars} variables, {model.NumConstrs} linear constraints, {model.NumQConstrs} quadratic constraints")

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print(f"Optimal value: {model.ObjVal:.2f}")
        for s in slots:
            for r_id in recipe_ids:
                v = assign[r_id, s].X
                if v > 1e-6:
                    print(f"  Slot {s}: {r_id} x{v:.4f}")
    else:
        print(f"No optimal solution found (status {model.Status})")


if __name__ == "__main__":
    main()

