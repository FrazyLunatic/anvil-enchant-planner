# Anvil Enchant Planner
A Minecraft Java Edition tool that figures out the most XP-efficient order to combine enchanted books and apply them to an item — without blowing past the 39-level anvil cap.

Hosted as a static site on GitHub Pages.

## Features
- Filters enchantments by item type
- Per-enchantment level selection
- Tracks Prior Work Penalty (PWP) for both items and books
- Optional rename cost (+1 level on the final step)
- Finds the cheapest valid anvil order automatically
- Steps that exceed 39 levels are flagged, but only as a last resort
- Full step-by-step breakdown of every anvil operation
- Generates `/give` commands (Java 1.20.5+ data components)
- Responsive UI with icons and tooltips

## How It Works
Each enchantment is treated as its own book. The solver runs dynamic programming over all possible combination orders to minimize total XP, preferring book-first combinations to slow down PWP growth.

**Prior Work Penalty formula:**
PWP(u) = 2^u − 1

where `u` is the number of prior anvil uses on that item or book.

## Project Structure
anvil-enchant-planner/
├─ index.html
├─ assets/
│  └─ Enchanted_Book.gif
└─ icons/
   ├─ favicon.ico
   ├─ favicon-16.png
   ├─ favicon-32.png
   └─ apple-touch-icon.png

## Running Locally
No build step needed — just open `index.html` in a browser, or use something like VS Code Live Server.

## Limitations
- One enchantment per book for now
- Java Edition only
- Targets modern anvil mechanics (1.20+)

## What's Next
- Multi-enchant books
- Bedrock Edition support
- XP vs level visualization
- Shareable permalink
- Mobile UI improvements

## Copyright
© Nick Nayak. All rights reserved.
