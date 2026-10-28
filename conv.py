from pathlib import Path

# Input file (original .mdpa or paste.txt)
in_path = Path("paste.txt")
text = in_path.read_text()

out_lines = []
scale = 1e-3
in_nodes = False

for line in text.splitlines():
    s = line.strip()

    if s.startswith("Begin Nodes"):
        in_nodes = True
        out_lines.append(line + "\n")
        continue

    if in_nodes:
        if s.startswith("End Nodes"):
            in_nodes = False
            out_lines.append(line + "\n")
            continue

        parts = line.split()
        # Expect lines like: "    id x y z"
        if len(parts) == 4 and parts[0].isdigit():
            nid = parts[0]
            x = float(parts[1]) * scale
            y = float(parts[2]) * scale
            z = float(parts[3]) * scale
            out_lines.append(
                f"    {nid} {x:.15g} {y:.15g} {z:.15g}\n"
            )
        else:
            # Keep any unexpected line inside the block as-is
            out_lines.append(line + "\n")
    else:
        out_lines.append(line + "\n")

out_path = Path("nodes_scaled_to_m.txt")
out_path.write_text("".join(out_lines))
print("written", out_path, "bytes", out_path.stat().st_size)