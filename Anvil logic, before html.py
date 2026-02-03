"""
UI flow:
1) Pick item
2) Pick enchants that apply + levels
3) Rename? (applied on final operation only; best practice)
4) Base item anvil uses
5) Books are assumed one-enchant-per-book (like your planning screenshot)
   Optionally enter uses per book (or all 0)

Output:
- Sum of base costs (sum of sacrificed-value costs across all operations)
- Sum of prior work penalties (sum of PWP(left)+PWP(right) across all operations)
- Total cost (levels) = base costs + PWP sum + rename_cost(0/1)
- A step-by-step combine plan
- If survival-valid plan not possible: prints LAST-RESORT plan and indicates which ops break 39.

Note:
- This version assumes each enchant appears in only one book (no duplicates across books).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import re

# -----------------------------
# Enchant max levels
# -----------------------------
ENCHANT_MAX: Dict[str, int] = {
    "Protection": 4, "Fire Protection": 4, "Blast Protection": 4, "Projectile Protection": 4,
    "Thorns": 3, "Unbreaking": 3, "Mending": 1, "Curse of Binding": 1, "Curse of Vanishing": 1,
    "Feather Falling": 4, "Depth Strider": 3, "Frost Walker": 2, "Soul Speed": 3,
    "Respiration": 3, "Aqua Affinity": 1, "Swift Sneak": 3,
    "Sharpness": 5, "Smite": 5, "Bane of Arthropods": 5, "Knockback": 2, "Fire Aspect": 2,
    "Looting": 3, "Sweeping Edge": 3,
    "Efficiency": 5, "Silk Touch": 1, "Fortune": 3,
    "Power": 5, "Punch": 2, "Flame": 1, "Infinity": 1,
    "Multishot": 1, "Piercing": 4, "Quick Charge": 3,
    "Impaling": 5, "Riptide": 3, "Loyalty": 3, "Channeling": 1,
    "Luck of the Sea": 3, "Lure": 3, "Wind Burst": 3, "Density": 5, "Breach": 4, "Lunge": 3,
}

# -----------------------------
# Cost multipliers (item, book) — from wiki table
# We use BOOK multipliers because we apply books to item and combine books.
# -----------------------------
ENCHANT_MULTIPLIERS: Dict[str, Tuple[int, int]] = {
    "Protection": (1, 1),
    "Fire Protection": (2, 1),
    "Feather Falling": (2, 1),
    "Blast Protection": (4, 2),
    "Projectile Protection": (2, 1),
    "Thorns": (8, 4),
    "Respiration": (4, 2),
    "Depth Strider": (4, 2),
    "Aqua Affinity": (4, 2),
    "Sharpness": (1, 1),
    "Smite": (2, 1),
    "Bane of Arthropods": (2, 1),
    "Knockback": (2, 1),
    "Fire Aspect": (4, 2),
    "Looting": (4, 2),
    "Efficiency": (1, 1),
    "Silk Touch": (8, 4),
    "Unbreaking": (2, 1),
    "Fortune": (4, 2),
    "Power": (1, 1),
    "Punch": (4, 2),
    "Flame": (4, 2),
    "Infinity": (8, 4),
    "Luck of the Sea": (4, 2),
    "Lure": (4, 2),
    "Frost Walker": (4, 2),
    "Mending": (4, 2),
    "Curse of Binding": (8, 4),
    "Curse of Vanishing": (8, 4),
    "Impaling": (4, 2),
    "Riptide": (4, 2),
    "Loyalty": (1, 1),
    "Channeling": (8, 4),
    "Multishot": (4, 2),
    "Piercing": (1, 1),
    "Quick Charge": (2, 1),
    "Soul Speed": (8, 4),
    "Swift Sneak": (8, 4),
    "Sweeping Edge": (4, 2),
    "Wind Burst": (4, 2),
    "Density": (2, 1),
    "Breach": (4, 2),
    "Lunge": (2, 1),
}

# -----------------------------
# Incompatibility groups (Java)
# -----------------------------
INCOMPAT_GROUPS: List[Set[str]] = [
    {"Protection", "Fire Protection", "Blast Protection", "Projectile Protection"},
    {"Sharpness", "Smite", "Bane of Arthropods"},
    {"Depth Strider", "Frost Walker"},
    {"Infinity", "Mending"},
    {"Multishot", "Piercing"},
    {"Loyalty", "Riptide"},
    {"Channeling", "Riptide"},
    {"Density", "Breach", "Smite", "Bane of Arthropods"},
    {"Sharpness", "Density", "Breach"},
]

def is_compatible(existing: Set[str], new_ench: str) -> bool:
    for grp in INCOMPAT_GROUPS:
        if new_ench in grp and any(e in grp for e in existing):
            return False
    return True

# -----------------------------
# Item -> allowed enchants (extend as needed)
# -----------------------------
ITEMS: Dict[str, Set[str]] = {
    "Helmet": {"Protection","Fire Protection","Blast Protection","Projectile Protection","Thorns","Respiration","Aqua Affinity","Unbreaking","Mending","Curse of Binding","Curse of Vanishing"},
    "Chestplate": {"Protection","Fire Protection","Blast Protection","Projectile Protection","Thorns","Unbreaking","Mending","Curse of Binding","Curse of Vanishing"},
    "Leggings": {"Protection","Fire Protection","Blast Protection","Projectile Protection","Thorns","Swift Sneak","Unbreaking","Mending","Curse of Binding","Curse of Vanishing"},
    "Boots": {"Protection","Fire Protection","Blast Protection","Projectile Protection","Feather Falling","Thorns","Depth Strider","Frost Walker","Soul Speed","Unbreaking","Mending","Curse of Binding","Curse of Vanishing"},
    "Sword": {"Sharpness","Smite","Bane of Arthropods","Knockback","Fire Aspect","Looting","Sweeping Edge","Unbreaking","Mending","Curse of Vanishing"},
    "Axe": {"Efficiency","Silk Touch","Fortune","Sharpness","Smite","Bane of Arthropods","Unbreaking","Mending","Curse of Vanishing"},
    "Pickaxe": {"Efficiency","Silk Touch","Fortune","Unbreaking","Mending","Curse of Vanishing"},
    "Shovel": {"Efficiency","Silk Touch","Fortune","Unbreaking","Mending","Curse of Vanishing"},
    "Hoe": {"Efficiency","Silk Touch","Fortune","Unbreaking","Mending","Curse of Vanishing"},
    "Bow": {"Power","Punch","Flame","Infinity","Unbreaking","Mending","Curse of Vanishing"},
    "Crossbow": {"Multishot","Piercing","Quick Charge","Unbreaking","Mending","Curse of Vanishing"},
    "Trident": {"Impaling","Riptide","Loyalty","Channeling","Unbreaking","Mending","Curse of Vanishing"},
    "Fishing Rod": {"Luck of the Sea","Lure","Unbreaking","Mending","Curse of Vanishing"},
    "Shield": {"Unbreaking","Mending","Curse of Vanishing"},
    "Elytra": {"Unbreaking","Mending","Curse of Vanishing"},
    "Shears": {"Efficiency","Unbreaking","Mending","Curse of Vanishing"},
    "Carrot on a Stick": {"Unbreaking","Mending","Curse of Vanishing"},
    "Warped Fungus on a Stick": {"Unbreaking","Mending","Curse of Vanishing"},
    "Spear": {"Sharpness", "Smite", "Bane of Arthropods", "Knockback", "Fire Aspect", "Looting", "Unbreaking", "Mending", "Curse of Vanishing","Lunge"},
    "Mace": {"Smite", "Bane of Arthropods", "Density", "Breach", "Wind Burst", "Fire Aspect", "Unbreaking", "Mending", "Curse of Vanishing"}

}

# -----------------------------
# Mechanics
# -----------------------------
def pwp(uses: int) -> int:
    return (1 << uses) - 1  # 2^uses - 1

def value_book(ench: Dict[str, int]) -> int:
    # sacrifice is a book (book multiplier)
    total = 0
    for name, lvl in ench.items():
        total += lvl * ENCHANT_MULTIPLIERS[name][1]
    return total

@dataclass(frozen=True)
class Book:
    ench: Dict[str, int]  # in this version: one enchant per book
    uses: int

@dataclass
class Op:
    kind: str  # "BOOK" or "ITEM"
    left_desc: str
    right_desc: str
    left_uses: int
    right_uses: int
    out_desc: str
    out_uses: int
    base_cost: int      # value(sacrifice)
    pwp_cost: int       # pwp(left)+pwp(right)
    rename_cost: int    # 0/1
    op_cost: int        # base + pwp + rename

@dataclass
class ParentPtr:
    left_mask: int
    left_uses: int
    right_mask: int
    right_uses: int
    renamed_here: bool

INF = 10**18

def solve_plan(
    books: List[Book],
    base_item_uses: int,
    rename: bool,
    enforce_survival: bool,  # True => op_cost<=39 required
) -> Tuple[int, int, int, int, List[Op]]:
    """
    Returns:
    (total_cost, sum_base_costs, sum_pwp_costs, final_item_uses, ops)
    """
    n = len(books)
    full = (1 << n) - 1
    MAXU = 20

    # Validate uniqueness + compatibility
    seen = set()
    for b in books:
        for e in b.ench:
            if e in seen:
                raise ValueError(f"Duplicate enchant '{e}' not supported yet in this version.")
            if not is_compatible(seen, e):
                raise ValueError(f"Incompatible set: adding '{e}' conflicts with existing selection.")
            seen.add(e)

    book_ench = [b.ench for b in books]
    book_uses = [b.uses for b in books]

    # Subset union + value
    subset_ench: List[Dict[str, int]] = [{} for _ in range(1 << n)]
    subset_val: List[int] = [0 for _ in range(1 << n)]
    for mask in range(1 << n):
        ench: Dict[str, int] = {}
        ok = True
        existing = set()
        for i in range(n):
            if mask & (1 << i):
                for e, lvl in book_ench[i].items():
                    if e in ench or not is_compatible(existing, e):
                        ok = False
                        break
                    ench[e] = lvl
                    existing.add(e)
            if not ok:
                break
        if ok:
            subset_ench[mask] = ench
            subset_val[mask] = value_book(ench)
        else:
            subset_val[mask] = INF

    # dp_book[mask][u] = (total_cost, base_sum, pwp_sum)
    dpb_cost = [[INF]*(MAXU+1) for _ in range(1 << n)]
    dpb_base = [[0]*(MAXU+1) for _ in range(1 << n)]
    dpb_pwp  = [[0]*(MAXU+1) for _ in range(1 << n)]
    parb: List[List[Optional[ParentPtr]]] = [[None]*(MAXU+1) for _ in range(1 << n)]

    for i in range(n):
        m = 1 << i
        u = book_uses[i]
        dpb_cost[m][u] = 0
        dpb_base[m][u] = 0
        dpb_pwp[m][u] = 0
        parb[m][u] = None

    # Build combined books
    for mask in range(1 << n):
        if mask == 0 or subset_val[mask] >= INF:
            continue
        sub = (mask - 1) & mask
        while sub:
            A = sub
            B = mask ^ A
            if B:
                for uA in range(MAXU+1):
                    if dpb_cost[A][uA] >= INF:
                        continue
                    for uB in range(MAXU+1):
                        if dpb_cost[B][uB] >= INF:
                            continue
                        u_out = max(uA, uB) + 1
                        if u_out > MAXU:
                            continue
                        base = subset_val[B]
                        pwp_cost = pwp(uA) + pwp(uB)
                        op_cost = base + pwp_cost
                        if enforce_survival and op_cost > 39:
                            continue
                        total = dpb_cost[A][uA] + dpb_cost[B][uB] + op_cost
                        if total < dpb_cost[mask][u_out]:
                            dpb_cost[mask][u_out] = total
                            dpb_base[mask][u_out] = dpb_base[A][uA] + dpb_base[B][uB] + base
                            dpb_pwp[mask][u_out]  = dpb_pwp[A][uA]  + dpb_pwp[B][uB]  + pwp_cost
                            parb[mask][u_out] = ParentPtr(A, uA, B, uB, False)
            sub = (sub - 1) & mask

    # dp_item[mask][u] for applying subsets to item
    dpi_cost = [[INF]*(MAXU+1) for _ in range(1 << n)]
    dpi_base = [[0]*(MAXU+1) for _ in range(1 << n)]
    dpi_pwp  = [[0]*(MAXU+1) for _ in range(1 << n)]
    pari: List[List[Optional[ParentPtr]]] = [[None]*(MAXU+1) for _ in range(1 << n)]
    dpi_cost[0][base_item_uses] = 0

    for mask in range(1 << n):
        for u_item in range(MAXU+1):
            if dpi_cost[mask][u_item] >= INF:
                continue
            remaining = full ^ mask
            sub = remaining
            while sub:
                B = sub
                for uB in range(MAXU+1):
                    if dpb_cost[B][uB] >= INF:
                        continue
                    u_out = max(u_item, uB) + 1
                    if u_out > MAXU:
                        continue
                    renamed_here = rename and ((mask | B) == full)
                    base = subset_val[B]
                    pwp_cost = pwp(u_item) + pwp(uB)
                    r = 1 if renamed_here else 0
                    op_cost = base + pwp_cost + r
                    if enforce_survival and op_cost > 39:
                        continue
                    new_mask = mask | B
                    total = dpi_cost[mask][u_item] + dpb_cost[B][uB] + op_cost
                    if total < dpi_cost[new_mask][u_out]:
                        dpi_cost[new_mask][u_out] = total
                        dpi_base[new_mask][u_out] = dpi_base[mask][u_item] + dpb_base[B][uB] + base
                        dpi_pwp[new_mask][u_out]  = dpi_pwp[mask][u_item]  + dpb_pwp[B][uB]  + pwp_cost
                        pari[new_mask][u_out] = ParentPtr(mask, u_item, B, uB, renamed_here)
                sub = (sub - 1) & remaining

    # Pick best finish
    best_cost = INF
    best_u = -1
    for u in range(MAXU+1):
        if dpi_cost[full][u] < best_cost:
            best_cost = dpi_cost[full][u]
            best_u = u
    if best_cost >= INF:
        raise RuntimeError("No plan found under current constraints.")

    # Reconstruct ops (postorder for book-building)
    def mask_idxs(mask: int) -> List[int]:
        return [i for i in range(n) if mask & (1 << i)]

    def book_desc_single(i: int) -> str:
        (e, lvl) = next(iter(book_ench[i].items()))
        return f"Book#{i}({e} {lvl})"

    ops: List[Op] = []
    built_cache: Dict[Tuple[int, int], str] = {}

    def emit_book(mask: int, uses: int) -> str:
        if mask & (mask - 1) == 0:
            i = mask.bit_length() - 1
            return book_desc_single(i)
        key = (mask, uses)
        if key in built_cache:
            return built_cache[key]

        p = parb[mask][uses]
        assert p is not None
        left_name = emit_book(p.left_mask, p.left_uses)
        right_name = emit_book(p.right_mask, p.right_uses)
        out_name = f"Book[{','.join(map(str, mask_idxs(mask)))}]"
        out_uses = max(p.left_uses, p.right_uses) + 1

        base = subset_val[p.right_mask]
        pwp_cost = pwp(p.left_uses) + pwp(p.right_uses)
        op_cost = base + pwp_cost

        ops.append(Op(
            kind="BOOK",
            left_desc=left_name, right_desc=right_name,
            left_uses=p.left_uses, right_uses=p.right_uses,
            out_desc=out_name, out_uses=out_uses,
            base_cost=base, pwp_cost=pwp_cost, rename_cost=0, op_cost=op_cost
        ))
        built_cache[key] = out_name
        return out_name

    # Walk item DP backwards
    cur_mask, cur_u = full, best_u
    chain: List[ParentPtr] = []
    while cur_mask != 0:
        p = pari[cur_mask][cur_u]
        assert p is not None
        chain.append(p)
        cur_mask, cur_u = p.left_mask, p.left_uses
    chain.reverse()

    item_uses = base_item_uses
    item_desc = "ITEM"

    for step in chain:
        B = step.right_mask
        uB = step.right_uses

        bname = emit_book(B, uB)  # emits needed BOOK ops first
        out_uses = max(item_uses, uB) + 1

        base = subset_val[B]
        pwp_cost = pwp(item_uses) + pwp(uB)
        r = 1 if step.renamed_here else 0
        op_cost = base + pwp_cost + r

        ops.append(Op(
            kind="ITEM",
            left_desc=item_desc, right_desc=bname,
            left_uses=item_uses, right_uses=uB,
            out_desc="ITEM", out_uses=out_uses,
            base_cost=base, pwp_cost=pwp_cost, rename_cost=r, op_cost=op_cost
        ))
        item_uses = out_uses

    total_cost = best_cost
    sum_base = dpi_base[full][best_u]
    sum_pwp_costs = dpi_pwp[full][best_u]
    return total_cost, sum_base, sum_pwp_costs, best_u, ops

# -----------------------------
# Terminal UI
# -----------------------------
ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
INT_TO_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}

def prompt_int(msg: str, lo: int, hi: int) -> int:
    while True:
        s = input(msg).strip()
        if not s.isdigit():
            print(f"Enter an integer {lo}-{hi}.")
            continue
        v = int(s)
        if lo <= v <= hi:
            return v
        print(f"Enter an integer {lo}-{hi}.")

def prompt_yesno(msg: str) -> bool:
    while True:
        s = input(msg + " [y/n]: ").strip().lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Enter y or n.")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def parse_selection_list(s: str, max_index: int) -> List[int]:
    s = normalize_spaces(s)
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out: Set[int] = set()
    for p in parts:
        if "-" in p:
            a, b = [x.strip() for x in p.split("-", 1)]
            if not (a.isdigit() and b.isdigit()):
                continue
            aa, bb = int(a), int(b)
            if aa > bb:
                aa, bb = bb, aa
            for k in range(aa, bb + 1):
                if 1 <= k <= max_index:
                    out.add(k)
        else:
            if p.isdigit():
                k = int(p)
                if 1 <= k <= max_index:
                    out.add(k)
    return sorted(out)

def prompt_level(enchant: str) -> int:
    mx = ENCHANT_MAX[enchant]
    while True:
        s = input(f"Level for {enchant} (1-{mx}, Enter=MAX): ").strip().upper()
        if s == "":
            return mx
        if s.isdigit():
            v = int(s)
            if 1 <= v <= mx:
                return v
        if s in ROMAN_TO_INT:
            v = ROMAN_TO_INT[s]
            if 1 <= v <= mx:
                return v
        print(f"Enter 1-{mx} (or roman I-V), or press Enter for max.")

def run():
    print("=== Anvil Planner (Java) — Survival-valid first ===\n")

    item_names = sorted(ITEMS.keys())
    print("What would you like to enchant?")
    for i, nm in enumerate(item_names, 1):
        print(f"{i:>2}. {nm}")
    idx = prompt_int("\nChoose item number: ", 1, len(item_names))
    item = item_names[idx - 1]

    allowed = sorted(ITEMS[item])
    print(f"\nEnchantments that apply to: {item}")
    for i, e in enumerate(allowed, 1):
        mx = ENCHANT_MAX.get(e, 1)
        mx_txt = INT_TO_ROMAN.get(mx, str(mx))
        print(f"{i:>2}. {e} (max {mx_txt})")

    print("\nSelect enchantments (e.g., 1,4,7 or 1-3,8):")
    chosen = []
    while not chosen:
        chosen = parse_selection_list(input("Enter selection: "), len(allowed))
        if not chosen:
            print("Select at least one enchant.")

    selected_names: Set[str] = set()
    selected: Dict[str, int] = {}
    print("")
    for i in chosen:
        ench = allowed[i - 1]
        if not is_compatible(selected_names, ench):
            conflict = []
            for grp in INCOMPAT_GROUPS:
                if ench in grp:
                    conflict = [x for x in grp if x in selected_names]
                    break
            print(f"Conflict: {ench} incompatible with {', '.join(conflict)}. Skipping.")
            continue
        lvl = prompt_level(ench)
        selected[ench] = lvl
        selected_names.add(ench)

    if not selected:
        print("\nAll chosen enchants conflicted. Re-run and pick a compatible set.")
        return

    rename = prompt_yesno("\nRename at the end?")

    base_uses = prompt_int("\nBase item anvil uses (0-20): ", 0, 20)

    all_fresh = prompt_yesno("\nAre all enchant books fresh (anvil uses = 0)?")
    books: List[Book] = []
    for name in sorted(selected.keys()):
        lvl = selected[name]
        u = 0
        if not all_fresh:
            u = prompt_int(f"Anvil uses for book '{name} {lvl}' (0-20): ", 0, 20)
        books.append(Book(ench={name: lvl}, uses=u))

    # 1) Try survival-valid plan (<=39 every op)
    try:
        total, base_sum, pwp_sum, final_uses, ops = solve_plan(
            books=books,
            base_item_uses=base_uses,
            rename=rename,
            enforce_survival=True,
        )
        survival_ok = True
    except Exception:
        survival_ok = False

    if not survival_ok:
        # 2) Last resort: best plan without the 39 cap
        total, base_sum, pwp_sum, final_uses, ops = solve_plan(
            books=books,
            base_item_uses=base_uses,
            rename=rename,
            enforce_survival=False,
        )

    print("\n=== SUMMARY ===")
    print("Item:", item)
    print("Enchant set:")
    for k in sorted(selected.keys()):
        v = selected[k]
        print(f" - {k} {INT_TO_ROMAN.get(v, str(v))}")
    print("Rename:", "Yes" if rename else "No")
    print("Base item uses:", base_uses)

    print("\nCosts:")
    print(" - Sum of base costs:", base_sum)
    print(" - Sum of prior work penalties:", pwp_sum)
    print(" - Total cost:", total)

    if survival_ok:
        print("\n=== BEST PLAN (SURVIVAL-VALID: never Too Expensive) ===")
    else:
        print("\n=== LAST-RESORT PLAN (some steps may be Too Expensive) ===")
        print("Reason: no ordering exists where every operation cost <= 39 with your inputs.\n")

    print("Do these anvil operations in order (left slot + right slot):\n")
    for i, op in enumerate(ops, 1):
        flag = ""
        if op.op_cost >= 40:
            flag = "  <-- TOO EXPENSIVE"
        print(
            f"{i:>2}. [{op.kind}] "
            f"{op.left_desc}(uses={op.left_uses})  +  {op.right_desc}(uses={op.right_uses})"
            f"  =>  {op.out_desc}(uses={op.out_uses})\n"
            f"    cost = base({op.base_cost}) + pwp({op.pwp_cost}) + rename({op.rename_cost}) = {op.op_cost}{flag}\n"
        )

if __name__ == "__main__":
    run()
