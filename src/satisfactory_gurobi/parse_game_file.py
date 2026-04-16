import json
import re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class Form(Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"


class Container(Enum):
    NEITHER = "neither"
    CANISTER = "canister"
    TANK = "tank"


# Items that are liquid/gas but use no container (transported via pipes only)
NO_CONTAINER_ITEMS = {"Desc_DissolvedSilica_C", "Desc_QuantumEnergy_C", "Desc_DarkEnergy_C"}

# Buildings not considered production facilities (exclude from producedIn)
NON_FACTORY_BUILDINGS = {"BP_WorkBenchComponent", "BP_WorkshopComponent", "BP_BuildGun",
                          "FGBuildableAutomatedWorkBench", "Build_AutomatedWorkBench"}

# Somersloop cost overrides (game data is sometimes incorrect or missing)
SOMERSLOOP_COST_OVERRIDES: dict[str, int] = {
    "Build_SmelterMk1_C": 1,
}

# Variable power buildings with fixed override costs
VARIABLE_POWER_COST: dict[str, float] = {
    "Build_QuantumEncoder_C": 1000.0,
    "Build_Converter_C": 250.0,
    "Build_HadronCollider_C": 1000.0,  # particle accelerator — actual cost depends on recipe
}

# Per-recipe power consumption overrides (MW)
RECIPE_POWER_OVERRIDES: dict[str, float] = {
    "Recipe_SpaceElevatorPart_10_C": 750.0,  # Biochemical Sculptor
}

# Raw resource item IDs
RAW_SOLID_RESOURCE_IDS = [
    "Desc_OreGold_C",      # Caterium Ore
    "Desc_OreCopper_C",    # Copper Ore
    "Desc_OreIron_C",      # Iron Ore
    "Desc_RawQuartz_C",    # Raw Quartz
    "Desc_OreBauxite_C",   # Bauxite
    "Desc_Stone_C",        # Limestone
    "Desc_Sulfur_C",       # Sulphur
    "Desc_OreUranium_C",   # Uranium
    "Desc_Coal_C",         # Coal
    "Desc_SAM_C",          # SAM
]
RAW_FLUID_RESOURCE_IDS = [
    "Desc_Water_C",        # Water
    "Desc_LiquidOil_C",    # Crude Oil
    "Desc_NitrogenGas_C",  # Nitrogen Gas
]
RAW_RESOURCE_IDS = RAW_SOLID_RESOURCE_IDS + RAW_FLUID_RESOURCE_IDS

# Per-minute output and power cost for raw extraction recipes
# Amounts are stored as (amount per 60s cycle) so rate = amount / 60 * 60 = amount/min
_SOLID_MINE_POWER = 151.1
_SOLID_MINE_AMOUNT = 1200  # /min
_FLUID_EXTRACT_POWER: dict[str, float] = {
    "Desc_Water_C":       67.2,
    "Desc_LiquidOil_C":  150.0,
    "Desc_NitrogenGas_C": 150.0,
}
_FLUID_EXTRACT_AMOUNT = 300_000  # /min

# Native class paths to parse
MANUFACTURER_CLASSES = {
    "FGBuildableManufacturer",
    "FGBuildableManufacturerVariablePower",
}
ITEM_CLASSES = {
    "FGItemDescriptor",
    "FGResourceDescriptor",
    "FGItemDescriptorBiomass",
    "FGItemDescriptorNuclearFuel",
    "FGItemDescriptorPowerBoosterFuel",
    "FGConsumableDescriptor",
}


@dataclass
class Buildable:
    id: str
    display_name: str
    somersloop_cost: int
    power_cost: float


@dataclass
class Item:
    id: str
    display_name: str
    form: Form
    container: Container
    value: int


@dataclass
class Recipe:
    id: str
    display_name: str
    ingredients: dict[str, float]
    products: dict[str, float]
    duration: float
    produced_in: str | None    # building id, or None if not a factory recipe
    power_consumption: float | None  # only set for particle accelerator recipes
    packaging: bool            # True if produced in the Packager
    raw: bool                  # True if this is a raw resource extraction recipe
    converter: bool            # True if this converts a raw resource via the Converter using Reanimated SAM
    somersloop_cost: int       # Number of somersloop slots the building uses (0 if not sloopable)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_item_amounts(raw: str) -> dict[str, float]:
    """Parse ingredient/product strings of the form:
    ((ItemClass="...Desc_IronIngot.Desc_IronIngot_C'",Amount=3),...)
    """
    result: dict[str, float] = {}
    for match in re.finditer(r'ItemClass="[^"]*?(\w+_C)[\'"]",Amount=(\d+)', raw):
        item_id = match.group(1)
        amount = float(match.group(2))
        result[item_id] = amount
    return result


def _parse_produced_in(raw: str) -> tuple[str | None, bool, bool]:
    """Return (primary_building_id, is_packaging, is_raw) from mProducedIn string."""
    building_ids = re.findall(r'(Build_\w+_C)', raw)
    # Filter to factory buildings only
    factory_buildings = [b for b in building_ids if b not in NON_FACTORY_BUILDINGS]

    is_packaging = "Build_Packager_C" in factory_buildings

    # Raw extraction: produced only by miners/extractors (none appear in FGRecipe)
    is_raw = False

    primary = factory_buildings[0] if factory_buildings else None
    return primary, is_packaging, is_raw


def _form(raw_form: str) -> Form:
    if "LIQUID" in raw_form:
        return Form.LIQUID
    if "GAS" in raw_form:
        return Form.GAS
    return Form.SOLID


def _container(item_id: str, form: Form) -> Container:
    if form == Form.SOLID:
        return Container.NEITHER
    if item_id in NO_CONTAINER_ITEMS:
        return Container.NEITHER
    if form == Form.LIQUID:
        return Container.CANISTER
    return Container.TANK


# ---------------------------------------------------------------------------
# Top-level parsers
# ---------------------------------------------------------------------------

def parse_buildables(data: list[dict]) -> list[Buildable]:
    buildables: list[Buildable] = []
    for section in data:
        nc = section["NativeClass"]
        if not any(mc in nc for mc in MANUFACTURER_CLASSES):
            continue
        is_variable = "VariablePower" in nc
        for cls in section["Classes"]:
            cls_id = cls["ClassName"]
            display_name = cls.get("mDisplayName", cls_id)
            somersloop_cost = SOMERSLOOP_COST_OVERRIDES.get(cls_id, int(cls.get("mProductionShardSlotSize", 0)))
            if is_variable:
                power_cost = VARIABLE_POWER_COST.get(cls_id, float(cls.get("mPowerConsumption", 0)))
            else:
                power_cost = float(cls.get("mPowerConsumption", 0))
            buildables.append(Buildable(
                id=cls_id,
                display_name=display_name,
                somersloop_cost=somersloop_cost,
                power_cost=power_cost,
            ))
    return buildables


def parse_items(data: list[dict]) -> tuple[list[Item], set[str]]:
    """Returns (items, raw_resource_ids)."""
    items: list[Item] = []
    raw_resource_ids: set[str] = set()
    for section in data:
        nc = section["NativeClass"]
        if not any(ic in nc for ic in ITEM_CLASSES):
            continue
        is_raw_resource = "FGResourceDescriptor" in nc
        for cls in section["Classes"]:
            cls_id = cls["ClassName"]
            display_name = cls.get("mDisplayName", cls_id)
            form = _form(cls.get("mForm", "RF_SOLID"))
            container = _container(cls_id, form)
            value = int(cls.get("mResourceSinkPoints", 0)) if form == Form.SOLID else 0
            items.append(Item(
                id=cls_id,
                display_name=display_name,
                form=form,
                container=container,
                value=value,
            ))
            if is_raw_resource:
                raw_resource_ids.add(cls_id)
    return items, raw_resource_ids


REANIMATED_SAM = "Desc_SAMIngot_C"


def parse_recipes(data: list[dict], raw_resource_ids: set[str], buildables: list[Buildable]) -> list[Recipe]:
    building_power = {b.id: b.power_cost for b in buildables}
    building_sloop = {b.id: b.somersloop_cost for b in buildables}
    recipes: list[Recipe] = []
    for section in data:
        if "FGRecipe'" not in section["NativeClass"]:
            continue
        for cls in section["Classes"]:
            cls_id = cls["ClassName"]
            display_name = cls.get("mDisplayName", cls_id)
            ingredients = _parse_item_amounts(cls.get("mIngredients", ""))
            products = _parse_item_amounts(cls.get("mProduct", ""))
            duration = float(cls.get("mManufactoringDuration", 0))
            produced_in_raw = cls.get("mProducedIn", "")
            produced_in, is_packaging, is_raw = _parse_produced_in(produced_in_raw)

            # Particle accelerator recipes have variable power from mVariablePowerConsumptionFactor;
            # all other factory recipes inherit their building's base power cost.
            # Per-recipe overrides take highest priority.
            power_factor = float(cls.get("mVariablePowerConsumptionFactor", 1.0))
            if cls_id in RECIPE_POWER_OVERRIDES:
                power_consumption = RECIPE_POWER_OVERRIDES[cls_id]
            elif produced_in == "Build_HadronCollider_C":
                power_consumption = power_factor
            else:
                power_consumption = building_power.get(produced_in)

            is_converter = (
                produced_in == "Build_Converter_C"
                and REANIMATED_SAM in ingredients
                and any(i in raw_resource_ids for i in ingredients if i != REANIMATED_SAM)
                and any(p in raw_resource_ids for p in products)
            )

            if produced_in is None:
                continue
            if any(kw in display_name for kw in ("Xmas", "Snow", "Candy Cane", "Ficsmas")):
                continue

            recipes.append(Recipe(
                id=cls_id,
                display_name=display_name,
                ingredients=ingredients,
                products=products,
                duration=duration,
                produced_in=produced_in,
                power_consumption=power_consumption,
                packaging=is_packaging,
                raw=is_raw,
                converter=is_converter,
                somersloop_cost=building_sloop.get(produced_in, 0),
            ))
    return recipes


def _make_raw_recipes(items: list[Item]) -> list[Recipe]:
    display_name = {item.id: item.display_name for item in items}
    recipes: list[Recipe] = []

    for item_id in RAW_SOLID_RESOURCE_IDS:
        recipes.append(Recipe(
            id=f"Mine_{item_id}",
            display_name=f"Mine {display_name.get(item_id, item_id)}",
            ingredients={},
            products={item_id: _SOLID_MINE_AMOUNT},
            duration=60,
            produced_in=None,
            power_consumption=_SOLID_MINE_POWER,
            packaging=False,
            raw=True,
            converter=False,
            somersloop_cost=0,
        ))

    for item_id in RAW_FLUID_RESOURCE_IDS:
        recipes.append(Recipe(
            id=f"Extract_{item_id}",
            display_name=f"Extract {display_name.get(item_id, item_id)}",
            ingredients={},
            products={item_id: _FLUID_EXTRACT_AMOUNT},
            duration=60,
            produced_in=None,
            power_consumption=_FLUID_EXTRACT_POWER[item_id],
            packaging=False,
            raw=True,
            converter=False,
            somersloop_cost=0,
        ))

    return recipes


def parse_game_file(path: Path) -> tuple[list[Buildable], list[Item], list[Recipe]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    buildables = parse_buildables(data)
    items, raw_resource_ids = parse_items(data)
    recipes = parse_recipes(data, raw_resource_ids, buildables) + _make_raw_recipes(items)
    return buildables, items, recipes

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


class _EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


if __name__ == "__main__":
    buildables, items, recipes = parse_game_file(DATA_DIR / "en-GB.json")

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "buildables.json", "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in buildables], f, indent=2, cls=_EnumEncoder)

    with open(OUTPUT_DIR / "items.json", "w", encoding="utf-8") as f:
        json.dump([asdict(i) for i in items], f, indent=2, cls=_EnumEncoder)

    with open(OUTPUT_DIR / "recipes.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in recipes], f, indent=2, cls=_EnumEncoder)

    print(f"Written {len(buildables)} buildables, {len(items)} items, {len(recipes)} recipes to {OUTPUT_DIR}")


