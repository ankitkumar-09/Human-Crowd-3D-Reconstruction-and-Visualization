import csv

csv_file = open("ground_points.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame_id", "person_id", "gx", "gy"])
frame_id = 0
frame_id += 1
for i, (gx, gy) in enumerate(people_ground):
    writer.writerow([frame_id, i, gx, gy])
csv_file.close()
