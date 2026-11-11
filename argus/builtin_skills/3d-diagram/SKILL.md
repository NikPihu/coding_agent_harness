---
name: 3d-diagram
description: >
  Make a 3D plot or diagram (surface, scatter, wireframe, shapes) and save it
  as a PNG image. Use when the user asks for a 3D chart, 3D visualization, or
  3D diagram of data or a math function.
---

# Making a 3D diagram

1. Write a standalone Python script to `diagrams/<short_name>.py` with
   write_file. Use matplotlib's 3D axes:

   ```python
   import matplotlib
   matplotlib.use("Agg")            # no window, just write the file
   import matplotlib.pyplot as plt
   import numpy as np

   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(projection="3d")
   # ... plot here: ax.plot_surface / ax.scatter / ax.plot_wireframe ...
   ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
   ax.set_title("<title>")
   fig.savefig("diagrams/<short_name>.png", dpi=150, bbox_inches="tight")
   print("saved diagrams/<short_name>.png")
   ```

2. Run it with uv, which brings the libraries along without installing them
   into the project:

   ```
   uv run --with matplotlib --with numpy python diagrams/<short_name>.py
   ```

3. If it fails, read the error, fix the script with str_replace, and run it
   again.

4. Tell the user the path of the PNG and one line on what it shows.
