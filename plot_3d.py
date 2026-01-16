import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1) Load data
pts = pd.read_csv("ground_points.csv")
risks = pd.read_csv("frame_risk.csv")

# 2) Choose which frame to visualize
frame_to_show = 10  # change this to any frame id

sub = pts[pts["frame_id"] == frame_to_show]
row = risks[risks["frame_id"] == frame_to_show]

if sub.empty or row.empty:
    print("No data for this frame_id")
    exit()

# 3) Decide color from risk (automatic)
risk = row["risk"].iloc[0]

if risk == "LOW":
    bar_color = "green"
elif risk == "MEDIUM":
    bar_color = "yellow"
else:
    bar_color = "red"

# 4) Prepare coordinates
x = sub["gx"].values
y = sub["gy"].values

z0 = [0.0] * len(x)   # ground level
dx = [0.3] * len(x)
dy = [0.3] * len(x)
dz = [1.7] * len(x)   # 1.7 m tall sticks

# 5) Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.bar3d(x, y, z0, dx, dy, dz, shade=True, color=bar_color)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Height (m)")
ax.set_title(f"Crowd 3D layout, frame {frame_to_show} (Risk: {risk})")

plt.show()
