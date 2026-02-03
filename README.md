# Anvil Enchant Planner

A **Minecraft Java Edition** anvil enchantment planner that computes the most XP-efficient order to combine enchanted books and apply them to an item, while attempting to keep **every step Survival-valid (≤39 levels)**.

Hosted as a **static site on GitHub Pages**.

## Features

- Item-specific enchantment filtering  
- Per-enchantment level selection  
- Tracks **Prior Work Penalty (PWP)** for items and books  
- Optional rename cost (+1 level on final step)  
- Automatically finds the cheapest valid anvil order  
- Marks steps that exceed 39 levels only as a **last resort**  
- Step-by-step breakdown of every anvil operation  
- Generates `/give` commands (Java 1.20.5+ data components)  
- Clean, responsive UI with icons and tooltips  

## How It Works

- Each enchantment is treated as **one book**
- Books are optimally combined using dynamic programming
- The solver minimizes total XP while respecting anvil mechanics
- Book-first combinations are preferred to reduce PWP growth

**Prior Work Penalty formula:**
PWP(u) = 2^u − 1
Where `u` is the number of prior anvil uses.

## Project Structure

anvil-enchant-planner/
├─ index.html
├─ assets/
│ └─ Enchanted_Book.gif
└─ icons/
├─ favicon.ico
├─ favicon-16.png
├─ favicon-32.png
└─ apple-touch-icon.png

## If you want to Run Locally

No build tools required.
1. Clone or download the repository
2. Open `index.html` in a browser  
   *(or use a local server like VS Code Live Server)*

## Limitations / Notes
  One enchantment per book (for now)
  Java Edition ruleset
  Designed for modern anvil mechanics (1.20+)

## Future Ideas to work on
  Multi-enchant books
  Bedrock Edition support
  XP vs level visualization
  Shareable permalink
  Mobile UI improvements

## Copyright
© Nick Nayak. All rights reserved.
cd anvil-enchant-planner
Then open index.html in a browser
(or use VS Code Live Server for best results).
